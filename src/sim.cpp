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
    registry.registerComponent<GrabData>();

    registry.registerSingleton<WorldReset>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<Material>();

    // Export tensors for pytorch
    registry.exportSingleton<WorldReset>(0);
    registry.exportColumn<Agent, MoveAction>(1);
    registry.exportColumn<Agent, Position>(2);

    registry.exportColumn<Material, Position>(3);
}

static void generateWorld(Engine &ctx)
{
    if (ctx.data().enableRender) {
        render::RenderingSystem::reset(ctx);
    }

    RigidBodyPhysicsSystem::reset(ctx);

    // Update the RNG seed for a new episode
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx =
        episode_mgr.curEpisode.fetch_add(1, std::memory_order_relaxed);
    ctx.data().rng = RNG::make(episode_idx);

    // Place the building materials
    // TODO - Add some randomness to this
    // TODO - Add scattered materials
    int materialsLen = 7;
    int materialsWidth = 7;
    ctx.data().numMaterials = materialsLen * materialsWidth;
    for (int i = 1; i <= materialsLen; i++) {
        for (int j = 1; j <= materialsWidth; j++) {
            Entity e = ctx.data().materials[i*materialsLen + j] = ctx.makeEntityNow<Material>();

            math::Vector3 pos {
                1 + (float)i * 2,
                1 + (float)j * 2,
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

    // Set building zone area
    ctx.data().zoneBoundsMinX = 16;
    ctx.data().zoneBoundsMaxX = 26;
    ctx.data().zoneBoundsMinY = 4;
    ctx.data().zoneBoundsMaxY = 14;

    // Set world bounds
    ctx.data().boundsX = 27;
    ctx.data().boundsY = 16;
    ctx.data().boundsZ = 10;

    ctx.getUnsafe<broadphase::LeafID>(ctx.data().plane) =
        RigidBodyPhysicsSystem::registerEntity(ctx, ctx.data().plane,
                                               ObjectID { 0 });

    // Reset the position of the agent
    Entity agent = ctx.data().agent;
    ctx.getUnsafe<Position>(agent) = Vector3 { 0, 0, 1 };
    ctx.getUnsafe<Rotation>(agent) = Quat { 1, 0, 0, 0 };
    ctx.getUnsafe<Velocity>(agent) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.getUnsafe<ExternalForce>(agent) = Vector3::zero();
    ctx.getUnsafe<ExternalTorque>(agent) = Vector3::zero();
    ctx.getUnsafe<broadphase::LeafID>(agent) =
        RigidBodyPhysicsSystem::registerEntity(ctx, agent, ObjectID { 1 });
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    if (!reset.resetNow) {
        return;
    }
    reset.resetNow = false;

    // Delete old materials
    const CountT num_materials = ctx.data().numMaterials;
    for (CountT i = 0; i < num_materials; i++) {
        ctx.destroyEntityNow(ctx.data().materials[i]);
    }

    generateWorld(ctx);
}

bool isInBuildingZone(Engine &ctx)
{
    Entity agent = ctx.data().agent;
    Vector3 agentPosition = ctx.getUnsafe<Position>(agent);

    return (agentPosition.x >= ctx.data().zoneBoundsMinX &&
            agentPosition.x <= ctx.data().zoneBoundsMaxX &&
            agentPosition.y >= ctx.data().zoneBoundsMinY &&
            agentPosition.y <= ctx.data().zoneBoundsMaxY);
}

void interactiveAction(Engine &ctx)
{
    Entity agent = ctx.data().agent;
    Vector3 cur_pos = ctx.getUnsafe<Position>(agent);
    Quat cur_rot = ctx.getUnsafe<Rotation>(agent);
    auto &grab_data = ctx.getUnsafe<GrabData>(agent);

    if (grab_data.constraintEntity != Entity::none()) {
        // Dropping object
        ctx.destroyEntityNow(grab_data.constraintEntity);
        grab_data.constraintEntity = Entity::none();
    }
    else {
        // Picking up object
        auto &bvh = ctx.getSingleton<broadphase::BVH>();
        float hit_t;
        Vector3 hit_normal;

        Vector3 ray_o = cur_pos - 0.5f * math::up;
        Vector3 ray_d = cur_rot.rotateVec(math::fwd);

        Entity grab_entity =
            bvh.traceRay(ray_o, ray_d, &hit_t, &hit_normal, 2.5f);

        if (grab_entity != Entity::none()) {
            auto &response_type = ctx.getUnsafe<ResponseType>(grab_entity);

            if (response_type == ResponseType::Dynamic) {

                Entity constraint_entity =
                    ctx.makeEntityNow<ConstraintData>();
                grab_data.constraintEntity = constraint_entity;

                Vector3 other_pos = ctx.getUnsafe<Position>(grab_entity);
                Quat other_rot = ctx.getUnsafe<Rotation>(grab_entity);

                Vector3 r1 = 1.25f * math::fwd - 0.5f * math::up;

                Vector3 hit_pos = ray_o + ray_d * hit_t;
                Vector3 r2 =
                    other_rot.inv().rotateVec(hit_pos - other_pos);

                Quat attach1 = { 1, 0, 0, 0 };
                Quat attach2 = (other_rot.inv() * cur_rot).normalize();

                float separation = hit_t - 1.25f;

                ctx.getUnsafe<JointConstraint>(constraint_entity) =
                    JointConstraint::setupFixed(agent, grab_entity,
                                                attach1, attach2,
                                                r1, r2, separation);

            }
        }
    }
}

inline void actionSystem(Engine &ctx,
                         MoveAction action,
                         const Rotation &agent_rot,
                         ExternalForce &agent_force,
                         ExternalTorque &agent_torque)
{
    // Check if at bounds
    Entity agent = ctx.data().agent;
    Vector3 curr_pos = ctx.getUnsafe<Position>(agent);

    // Translate from discrete actions to forces
    switch (action) {
    case MoveAction::Wait: {
        // Do nothing
    } break;
    case MoveAction::Left: {
        agent_force -= agent_rot.rotateVec(math::right);
    } break;
    case MoveAction::Right: {
        agent_force += agent_rot.rotateVec(math::right);
    } break;
    case MoveAction::Forward: {
        agent_force += agent_rot.rotateVec(math::fwd);
    } break;
    case MoveAction::Backward: {
        agent_force -= agent_rot.rotateVec(math::fwd);
    } break;
    case MoveAction::TurnLeft: {
        agent_torque -= agent_rot.rotateVec(math::up);
    } break;
    case MoveAction::TurnRight: {
        agent_torque += agent_rot.rotateVec(math::up);
    } break;
    case MoveAction::Jump: {
        // TODO - Add logic for jumping
    } break;
    case MoveAction::Interact: {
        interactiveAction(ctx);
    } break;
    default: __builtin_unreachable();
    }

    // Check if falling outside the world's bounds
    if (curr_pos.x < 0 || curr_pos.y < 0 ||  curr_pos.x >= ctx.data().boundsX ||
            curr_pos.y >= ctx.data().boundsY || curr_pos.z >= ctx.data().boundsZ) {
        agent_force = Vector3 { 0, 0, 0 };
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
    auto sort_material = queueSortByWorld<Material>(builder, {reset_sys});
    auto sort_agent = queueSortByWorld<Material>(builder, {sort_material});

    auto post_reset = sort_agent;
#endif

    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
        MoveAction, Rotation, ExternalForce, ExternalTorque>>({post_reset});

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
    if (ctx.data().enableRender) {
        render::RenderingSystem::reset(ctx);
    }
    int maxMaterialsLen = 9;
    int maxMaterialsWidth = 9;

    int maxMaterials = maxMaterialsLen * maxMaterialsWidth;

    RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr, deltaT,
                                 numPhysicsSubsteps, -9.8 * math::up,
                                 maxMaterials + 2,
                                 maxMaterials * maxMaterials / 2,
                                 10);

    // Allocate enough to hold max number of materials
    // TODO - Add extra for materials scattered
    materials = (Entity *)rawAlloc(sizeof(Entity) * maxMaterialsLen * maxMaterialsWidth);

    // Create ground plane during initialization, we reuse Material for this
    plane = ctx.makeEntityNow<Material>();
    ctx.getUnsafe<Position>(plane) = Vector3::zero();
    ctx.getUnsafe<Rotation>(plane) = Quat { 1, 0, 0, 0 };
    ctx.getUnsafe<Scale>(plane) = Diag3x3 { 1, 1, 1 };
    ctx.getUnsafe<ObjectID>(plane) = ObjectID { 0 };
    ctx.getUnsafe<ResponseType>(plane) = ResponseType::Static;

    // Generate agent
    agent = ctx.makeEntityNow<Agent>();
    ctx.getUnsafe<MoveAction>(agent) = MoveAction::Wait;
    ctx.getUnsafe<Scale>(agent) = Diag3x3 { 1, 1, 1 };
    ctx.getUnsafe<ObjectID>(agent) = ObjectID { 1 };
    ctx.getUnsafe<ResponseType>(agent) = ResponseType::Dynamic;
    ctx.getUnsafe<render::ViewSettings>(agent) =
        render::RenderingSystem::setupView(ctx, 90.f, 0.001f,
                                           math::up * 0.5f, { 0 });

    enableRender = cfg.enableRender;

    generateWorld(ctx);

    ctx.getSingleton<WorldReset>().resetNow = false;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
