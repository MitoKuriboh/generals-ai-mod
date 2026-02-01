/*
**	Command & Conquer Generals Zero Hour(tm)
**	Copyright 2025 Electronic Arts Inc.
**
**	This program is free software: you can redistribute it and/or modify
**	it under the terms of the GNU General Public License as published by
**	the Free Software Foundation, either version 3 of the License, or
**	(at your option) any later version.
**
**	This program is distributed in the hope that it will be useful,
**	but WITHOUT ANY WARRANTY; without even the implied warranty of
**	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**	GNU General Public License for more details.
**
**	You should have received a copy of the GNU General Public License
**	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  Tactical State - State representation for tactical layer ML inference    //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

// TacticalState.h
// Team-level state for tactical decisions
// Author: Mito, 2025

#pragma once

#ifndef _TACTICAL_STATE_H_
#define _TACTICAL_STATE_H_

#include "Common/GameMemory.h"
#include "Common/GameType.h"

// Forward declarations
class Team;
class Object;
class Player;

// =============================================================================
// Tactical Configuration Constants
// =============================================================================
namespace TacticalConfig {
    // Decision timing (in frames, 30 FPS)
    static const UnsignedInt DECISION_INTERVAL = 150;  // 5 seconds per team

    // State dimensions
    static const UnsignedInt STATE_DIM = 64;

    // Spatial calculations
    static const Real QUADRANT_RADIUS = 800.0f;        // Detection radius for quadrant analysis
    static const Real MAX_OBJECTIVE_DIST = 2000.0f;    // Max distance for normalization
    static const Real MAX_BASE_DIST = 3000.0f;         // Max base distance

    // Team thresholds
    static const Real LOW_HEALTH_THRESHOLD = 0.3f;     // Health ratio for "retreat needed"
    static const Real HIGH_THREAT_THRESHOLD = 0.7f;    // Threat level for danger
}

/**
 * Tactical action types corresponding to Python TacticalAction enum.
 */
enum TacticalActionType
{
    TACTICAL_ATTACK_MOVE = 0,     // Attack-move to position
    TACTICAL_ATTACK_TARGET = 1,   // Focus on specific target
    TACTICAL_DEFEND_POSITION = 2, // Guard location
    TACTICAL_RETREAT = 3,         // Fall back to base
    TACTICAL_HOLD = 4,            // Hold position
    TACTICAL_HUNT = 5,            // Seek and destroy
    TACTICAL_REINFORCE = 6,       // Merge with another team
    TACTICAL_SPECIAL = 7,         // Use special ability

    TACTICAL_ACTION_COUNT = 8
};

/**
 * Tactical state for team-level ML inference.
 * 64 dimensions matching the Python TacticalState.
 */
struct TacticalState
{
    // From strategic layer (8 floats)
    Real strategyEmbedding[8];

    // Team composition (12 floats) - [count, health, ready] x 4
    Real teamInfantry[3];     // Infantry units
    Real teamVehicles[3];     // Vehicle units
    Real teamAircraft[3];     // Air units
    Real teamMixed[3];        // Mixed/other units

    // Team status (8 floats)
    Real teamHealth;          // Average health ratio 0-1
    Real ammunition;          // Normalized ammo status
    Real cohesion;            // How spread out the team is (0=scattered, 1=tight)
    Real experience;          // Average veterancy level
    Real distToObjective;     // Distance to current objective (normalized)
    Real distToBase;          // Distance to home base (normalized)
    Real underFire;           // 1.0 if taking damage, else 0
    Real hasTransport;        // 1.0 if team has transport available

    // Situational (16 floats)
    Real nearbyEnemies[4];    // Enemy presence in 4 quadrants (NE, SE, SW, NW)
    Real nearbyAllies[4];     // Ally presence in 4 quadrants
    Real terrainAdvantage;    // Height/cover advantage
    Real threatLevel;         // Estimated danger level
    Real targetValue;         // Value of current target
    Real supplyDist;          // Distance to nearest supply
    Real retreatPath;         // Quality of retreat path (0=blocked, 1=clear)
    Real reinforcePossible;   // 1.0 if reinforcements available
    Real specialReady;        // 1.0 if special ability ready
    Real padding1;            // Reserved

    // Current objective (8 floats)
    Real objectiveType;       // Encoded objective type
    Real objectiveX;          // Normalized X position (0-1)
    Real objectiveY;          // Normalized Y position (0-1)
    Real priority;            // Objective priority (0-1)
    Real progress;            // Progress toward objective (0-1)
    Real timeOnObjective;     // Time spent on current objective (normalized)
    Real padding2;            // Reserved
    Real padding3;            // Reserved

    // Temporal (4 floats)
    Real timeSinceEngagement; // Time since last combat
    Real timeSinceCommand;    // Time since last command
    Real framesSinceSpawn;    // How long team has existed
    Real padding4;            // Reserved

    // Additional padding for 64-dim alignment (8 floats)
    Real padding5;
    Real padding6;
    Real padding7;
    Real padding8;
    Real padding9;
    Real padding10;
    Real padding11;
    Real padding12;

    // Initialize to zeros
    void clear();

    // Serialize to float array for JSON transmission
    void toFloatArray(Real* outArray) const;

    // State dimension
    static const UnsignedInt DIM = 64;
};

/**
 * Tactical command from ML system.
 */
struct TacticalCommand
{
    TacticalActionType action;  // Discrete action 0-7
    Real targetX;               // Target X position (0-1, map normalized)
    Real targetY;               // Target Y position (0-1, map normalized)
    Real attitude;              // Aggression level (0=passive, 1=aggressive)

    // Is this command valid?
    Bool valid;

    // Team ID this command is for
    Int teamId;

    void clear();
};

/**
 * Build tactical state from a Team object.
 *
 * @param team The team to extract state from
 * @param strategicOutput Current strategic layer output (8 floats)
 * @param player The owning player (for calculating base position etc.)
 * @param outState Output state struct
 */
void buildTacticalState(
    Team* team,
    const Real* strategicOutput,
    Player* player,
    TacticalState& outState
);

/**
 * Get team category weight from composition.
 * Returns (infantry, vehicles, aircraft, mixed) weights.
 */
void getTeamCompositionWeights(Team* team, Real* outWeights);

/**
 * Calculate team health as average of member health ratios.
 */
Real calculateTeamHealth(Team* team);

/**
 * Calculate team cohesion (how spread out units are).
 */
Real calculateTeamCohesion(Team* team);

/**
 * Count enemies in each quadrant around team center.
 * Fills 4-float array with normalized enemy presence.
 */
void countEnemiesInQuadrants(Team* team, Player* player, Real* outQuadrants);

/**
 * Count allies in each quadrant around team center.
 */
void countAlliesInQuadrants(Team* team, Player* player, Real* outQuadrants);

/**
 * Calculate threat level for team's position.
 */
Real calculateThreatLevel(Team* team, Player* player);

#endif // _TACTICAL_STATE_H_
