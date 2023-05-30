#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace SimpleExample {

constexpr inline float deltaT = 1.f / 30.f;
constexpr inline CountT numPhysicsSubsteps = 1;

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);
    RigidBodyPhysicsSystem::registerTypes(registry);
    render::RenderingSystem::registerTypes(registry);

    registry.registerComponent<MoveAction>();

    registry.registerSingleton<WorldReset>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<Zone>();
    registry.registerArchetype<Material>();

    // Export tensors for pytorch
    registry.exportSingleton<WorldReset>(0);
    registry.exportColumn<Agent, MoveAction>(1);
    registry.exportColumn<Agent, Position>(2);
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    if (!reset.resetNow) {
        return;
    }
    reset.resetNow = false;

    // TODO - reset agent position

    // TODO - reset materials
}

inline void actionSystem(Engine &,
                         MoveAction action,
                         const Rotation &agent_rot,
                         ExternalForce &agent_force)
{
    // Translate from discrete actions to forces
    switch (action) {
    case MoveAction::Wait: {
        // Do nothing
    } break;
    case MoveAction::Forward: {
        agent_force += agent_rot.rotateVec(math::fwd);
    } break;
    case MoveAction::Left: {
        agent_force -= agent_rot.rotateVec(math::right);
    } break;
    case MoveAction::Right: {
        agent_force += agent_rot.rotateVec(math::right);
    } break;
    default: __builtin_unreachable();
    }
}

#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraph::Builder &builder,
                                   Span<const TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
                                    deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}
#endif

void Sim::setupTasks(TaskGraph::Builder &builder, const Config &cfg)
{
    auto reset_sys =
        builder.addToGraph<ParallelForNode<Engine, resetSystem, WorldReset>>({});


#ifndef MADRONA_GPU_MODE
    auto post_reset = reset_sys;
#else
    // in GPU mode, need to queue up systems that sort and compact the ECS
    // tables in order to reclaim memory
    auto sort_obstacles = queueSortByWorld<Obstacle>(builder, {reset_sys});
    auto sort_agent = queueSortByWorld<Obstacle>(builder, {sort_obstacles});

    auto post_reset = sort_agent;
#endif

    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
        MoveAction, Rotation, ExternalForce>>({post_reset});

    auto bvh_sys = RigidBodyPhysicsSystem::setupBroadphaseTasks(
        builder, {action_sys});

    auto substep_sys = RigidBodyPhysicsSystem::setupSubstepTasks(
        builder, {bvh_sys}, numPhysicsSubsteps);

    auto phys_cleanup_sys = RigidBodyPhysicsSystem::setupCleanupTasks(
        builder, {substep_sys});

    auto sim_done = phys_cleanup_sys;
    if (cfg.enableRender) {
        sim_done = render::RenderingSystem::setupTasks(builder, {sim_done});
    }

#ifdef MADRONA_GPU_MODE
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({sim_done});
    (void)recycle_sys;
#endif
}


Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr)
{
    // TODO - initialize physic system
    // RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr, deltaT,
    //                              numPhysicsSubsteps, -9.8 * math::up,
    //                              init.numObstacles + 2,
    //                              init.numObstacles * init.numObstacles / 2,
    //                              10);

    agent = ctx.makeEntityNow<Agent>();
    ctx.getUnsafe<Position>(agent) = Vector3 { 0, 0, 1 };
    ctx.getUnsafe<Rotation>(agent) = Quat { 1, 0, 0, 0 };
    ctx.getUnsafe<Velocity>(agent) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.getUnsafe<ExternalForce>(agent) = Vector3::zero();
    ctx.getUnsafe<ExternalTorque>(agent) = Vector3::zero();
    ctx.getUnsafe<MoveAction>(agent) = MoveAction::Wait;
    ctx.getUnsafe<Scale>(agent) = Diag3x3 { 1, 1, 1 };
    ctx.getUnsafe<ObjectID>(agent) = ObjectID { 1 };
    ctx.getUnsafe<ResponseType>(agent) = ResponseType::Dynamic;
    ctx.getUnsafe<render::ViewSettings>(agent) =
        render::RenderingSystem::setupView(ctx, 90.f, 0.001f,
                                           math::up * 0.5f, { 0 });
    ctx.getUnsafe<broadphase::LeafID>(agent) =
        RigidBodyPhysicsSystem::registerEntity(ctx, agent, ObjectID { 1 });

    // Create ground plane during initialization
    plane = ctx.makeEntityNow<Zone>();
    ctx.getUnsafe<Position>(plane) = Vector3::zero();
    ctx.getUnsafe<Rotation>(plane) = Quat { 1, 0, 0, 0 };
    ctx.getUnsafe<Scale>(plane) = Diag3x3 { 1, 1, 1 };
    ctx.getUnsafe<ObjectID>(plane) = ObjectID { 0 };
    ctx.getUnsafe<ResponseType>(plane) = ResponseType::Static;

    // Generate building zone
    buildZone = ctx.makeEntityNow<Zone>();

    math::Vector3 zonePos {
            4,
            4,
            3,
    };

    ctx.getUnsafe<Position>(buildZone) = zonePos;
    ctx.getUnsafe<Rotation>(buildZone) = Quat { 1, 0, 0, 0 };
    ctx.getUnsafe<Scale>(buildZone) = Diag3x3 { 6, 6, 6 };
    ctx.getUnsafe<ObjectID>(buildZone) = ObjectID { 0 };
    ctx.getUnsafe<ResponseType>(buildZone) = ResponseType::Static;

    // Generate materials
    int materialsLen = 7;
    int materialsWidth = 7;

    materials = (Entity *)rawAlloc(sizeof(Entity) * materialsLen * materialsWidth);
    enableRender = cfg.enableRender;

    for (int i = 1; i <= materialsLen; i++) {
        for (int j = 1; j <= materialsWidth; j++) {
            Entity e = ctx.data().materials[i*materialsLen + j] = ctx.makeEntityNow<Material>();

            math::Vector3 pos {
                1 + (float)i,
                1 + (float)j,
                1.f,
            };

            ctx.getUnsafe<Position>(e) = pos;
            ctx.getUnsafe<Rotation>(e) = Quat { 1, 0, 0, 0 };
            ctx.getUnsafe<Scale>(e) = Diag3x3 { 1, 1, 1 };
            ctx.getUnsafe<Velocity>(e) = {
                Vector3::zero(),
                Vector3::zero(),
            };
            ctx.getUnsafe<ObjectID>(e) = ObjectID { 1 };
            ctx.getUnsafe<ResponseType>(e) = ResponseType::Dynamic;
            ctx.getUnsafe<ExternalForce>(e) = Vector3::zero();
            ctx.getUnsafe<ExternalTorque>(e) = Vector3::zero();
            ctx.getUnsafe<broadphase::LeafID>(e) =
                RigidBodyPhysicsSystem::registerEntity(ctx, e, ObjectID { 1 });
        }
    }

    ctx.getSingleton<WorldReset>().resetNow = false;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
