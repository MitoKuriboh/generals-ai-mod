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
//  Learning AI Player - ML-driven strategic decisions                        //
//  Extends AISkirmishPlayer with machine learning capabilities               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

// AILearningPlayer.h
// ML-driven computerized opponent
// Author: Mito, 2025

#pragma once

#ifndef _AI_LEARNING_PLAYER_H_
#define _AI_LEARNING_PLAYER_H_

#include "Common/GameMemory.h"
#include "GameLogic/AISkirmishPlayer.h"
#include "GameLogic/MLBridge.h"

class BuildListInfo;
class SpecialPowerTemplate;
class TeamPrototype;
class Team;
class AIGroup;

// =============================================================================
// ML Configuration Constants
// =============================================================================
namespace MLConfig {
    // Decision timing
    // Game runs at 30 FPS, so 30 frames = 1 second between ML decisions
    // This balances responsiveness with avoiding decision spam
    static const UnsignedInt DECISION_INTERVAL = 30;
    static const Real MAX_ATTACK_HOLD_SECONDS = 30.0f;         // Max time to hold teams before attacking

    // Spatial calculations
    static const Real THREAT_DETECTION_RADIUS = 500.0f;        // Base threat detection radius
    static const Real NORMALIZED_MAP_SCALE = 3000.0f;          // Map scale for distance normalization
    static const Real MAX_ARMY_STRENGTH_RATIO = 2.0f;          // Cap for army strength ratio

    // Aggression thresholds
    static const Real AGGRESSION_DEFENSIVE_THRESHOLD = 0.3f;   // Below this = defensive
    static const Real AGGRESSION_AGGRESSIVE_THRESHOLD = 0.7f;  // Above this = aggressive

    // Building priorities
    static const Real MIN_PRIORITY_WEIGHT = 0.1f;              // Minimum weight for any option
    static const Real DELAY_THRESHOLD = 0.15f;                 // Below this = delay building
}

// Team type classification for ML-influenced selection
enum TeamCategory
{
	TEAM_CATEGORY_INFANTRY,
	TEAM_CATEGORY_VEHICLE,
	TEAM_CATEGORY_AIRCRAFT,
	TEAM_CATEGORY_MIXED,
	TEAM_CATEGORY_UNKNOWN
};

// Building type classification for ML-influenced priorities
enum BuildingCategory
{
	BUILDING_CATEGORY_ECONOMY,    // Supply, refineries, etc.
	BUILDING_CATEGORY_POWER,      // Power plants
	BUILDING_CATEGORY_DEFENSE,    // Turrets, walls
	BUILDING_CATEGORY_MILITARY,   // Barracks, war factory, airfield
	BUILDING_CATEGORY_TECH,       // Tech center, strategy center
	BUILDING_CATEGORY_SUPER,      // Superweapons
	BUILDING_CATEGORY_UNKNOWN
};

/**
 * ML-driven computer-controlled opponent.
 * Extends AISkirmishPlayer to use machine learning for strategic decisions
 * while leveraging the existing infrastructure for command execution.
 */
class AILearningPlayer : public AISkirmishPlayer
{
	MEMORY_POOL_GLUE_WITH_USERLOOKUP_CREATE( AILearningPlayer, "AILearningPlayer" )

public:
	AILearningPlayer( Player *p );
	// Note: destructor declared by MEMORY_POOL_GLUE macro

	// AIPlayer interface methods
	virtual void update();
	virtual void newMap();
	virtual void onUnitProduced( Object *factory, Object *unit );

	// Identify as learning AI
	virtual Bool isLearningAI(void) { return true; }

protected:
	// Snapshot methods
	virtual void crc( Xfer *xfer );
	virtual void xfer( Xfer *xfer );
	virtual void loadPostProcess( void );

	// Decision methods - override for ML
	virtual Bool selectTeamToBuild( void );
	virtual void processBaseBuilding( void );
	virtual void processTeamBuilding( void );

	// Override to control attack timing based on aggression
	virtual void checkReadyTeams( void );

private:
	// ML Bridge communication
	void exportStateToML();
	void buildGameState(MLGameState& state);
	void processMLRecommendations();

	// Helper methods for state extraction
	void countForces(Real* infantry, Real* vehicles, Real* aircraft, Real* structures, Bool own);
	Real calculateTechLevel();
	Real calculateBaseThreat();
	Real calculateArmyStrength();
	Real calculateDistanceToEnemy();

	// Team classification helpers
	TeamCategory classifyTeam(TeamPrototype* proto);
	Real getTeamCategoryWeight(TeamCategory category);

	// Building classification helpers
	BuildingCategory classifyBuilding(const ThingTemplate* tmpl);
	Real getBuildingCategoryWeight(BuildingCategory category);
	Bool shouldDelayBuilding(BuildingCategory category);

	// Attack timing based on aggression
	Int getMinArmySizeForAttack();

	// Check for game end and notify ML
	void checkGameEnd();

	// ========================================================================
	// Hierarchical Control - Tactical and Micro layers
	// ========================================================================

	// Process tactical decisions for all active teams
	void processTeamTactics();

	// Process micro decisions for units in combat
	void processMicroControl();

	// Build tactical state for a team
	void buildTeamTacticalState(Team* team, TacticalState& outState);

	// Execute tactical command on a team
	void executeTacticalCommand(Team* team, const TacticalCommand& cmd);

	// Build micro state for a unit
	void buildUnitMicroState(Object* unit, MicroState& outState);

	// Execute micro command on a unit
	void executeMicroCommand(Object* unit, const MicroCommand& cmd);

	// Collect teams that need tactical updates for batched request
	void collectTeamsForBatch(MLBatchedRequest& request);

	// Collect units that need micro updates for batched request
	void collectUnitsForBatch(MLBatchedRequest& request);

	// Check if a unit should receive micro control
	Bool shouldMicroUnit(Object* unit);

	// Check if a team needs tactical update this frame
	Bool teamNeedsTacticalUpdate(Team* team);

	// Check if a unit needs micro control this frame
	Bool unitNeedsMicroUpdate(Object* unit);

	// Get base position for retreat calculations
	Bool getBasePosition(Coord3D& outPos);

	// ML Bridge instance
	MLBridge m_mlBridge;

	// Current ML recommendation
	MLRecommendation m_currentRecommendation;

	// State tracking
	UnsignedInt m_frameCounter;

	// Income tracking for state extraction
	Real m_lastFrameMoney;     // Money from previous frame for income calculation
	Real m_recentDamageTaken;  // Damage taken recently (for underAttack flag)
	UnsignedInt m_lastDamageFrame;  // Frame when damage was last taken

	// Attack timing state
	Int m_teamsHeld;                  // Number of teams waiting to attack
	UnsignedInt m_lastAttackFrame;    // Frame when last attack was launched

	// Game end tracking
	Bool m_gameEndSent;        // True if we've sent game end notification

	// Helper methods for complete state extraction
	Real calculateIncomeRate();
	Real calculateSupplyUsed();
	Real calculateUnderAttack();

	// ========================================================================
	// Hierarchical Control State
	// ========================================================================

	// Tactical layer configuration
	Bool m_tacticalEnabled;                 // Enable tactical layer
	UnsignedInt m_tacticalDecisionInterval; // Frames between tactical updates

	// Micro layer configuration
	Bool m_microEnabled;                    // Enable micro layer
	UnsignedInt m_microDecisionInterval;    // Frames between micro updates

	// Per-team tracking (using simple arrays since team count is bounded)
	static const Int MAX_TRACKED_TEAMS = 32;
	UnsignedInt m_lastTacticalFrame[MAX_TRACKED_TEAMS];  // Frame of last tactical update per team

	// Per-unit tracking (using map for dynamic unit IDs)
	// Note: In a real implementation, would use a more efficient structure
	static const Int MAX_TRACKED_UNITS = 128;
	ObjectID m_trackedUnitIds[MAX_TRACKED_UNITS];
	UnsignedInt m_lastMicroFrame[MAX_TRACKED_UNITS];
	Int m_trackedUnitCount;

	// Cached strategic output for tactical layer
	Real m_strategicOutput[8];

	// Base position cache
	Coord3D m_cachedBasePos;
	Bool m_hasValidBasePos;
};

#endif // _AI_LEARNING_PLAYER_H_
