#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>
#include <madrona/physics.hpp>
#include <madrona/mw_render.hpp>

#include "init.hpp"
#include "rng.hpp"

namespace SimpleExample {

// 3D Position & Quaternion Rotation
// These classes are defined in include/madrona/components.hpp
using madrona::Entity;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;
using madrona::phys::Velocity;
using madrona::phys::ResponseType;
using madrona::phys::ExternalForce;
using madrona::phys::ExternalTorque;

class Engine;

struct WorldReset {
    int32_t resetNow;
};

enum class MoveAction : int32_t {
    Wait = 0,
    Left = 1,
    Right = 2,
    Forward = 3,
    Backward = 4,
    TurnLeft = 5,
    TurnRight = 6,
    Jump = 7,
    Interact = 8
};

struct GrabData {
    Entity constraintEntity;
};

struct Reward {
        float rew;
};


enum class CarryingObj : int32_t {
    NotCarrying = 0,
    IsCarrying = 1
};

struct Material : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID
> {};

// Same as Obstacle but with an additional action and camera components
struct Agent : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,
    MoveAction,
    GrabData,
    madrona::render::ViewSettings
> {};

struct Sim : public madrona::WorldBase {
    struct Config {
        bool enableRender;
    };

    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraph::Builder &builder,
                           const Config &cfg);

    Sim(Engine &ctx, const Config &cfg, const WorldInit &init);

    EpisodeManager *episodeMgr;
    RNG rng;

    madrona::Entity agent;
    madrona::Entity plane;

    madrona::Entity *materials;
    int32_t numMaterials;
    int32_t numAgents;

    int32_t boundsX;
    int32_t boundsY;
    int32_t boundsZ;

    // Build zone bounds
    int32_t zoneBoundsMinX;
    int32_t zoneBoundsMinY;

    // Build zone bounds
    int32_t zoneBoundsMaxX;
    int32_t zoneBoundsMaxY;

    bool enableRender;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
