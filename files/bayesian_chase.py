"""
Real-time grid-based stealth / pursuit demo with Bayesian state estimation
=========================================================================

Overview
--------
This program simulates a simple 2D grid world with:
    - A player controlled in real time via WASD.
    - An enemy that tries to locate and chase the player.
    - A probabilistic belief state (a Bayesian filter) maintained by the enemy
      about the player’s possible location.

The world is a HEIGHT x WIDTH grid. On each game "tick" (a fixed time step),
the enemy updates its belief about where the player might be, then selects a
one-step move based on that belief or direct visual information.

Key Concepts and Data Structures
--------------------------------
1. Grid and positions
   - The grid is defined by:
         HEIGHT, WIDTH
   - Positions are represented as (row, col) tuples:
         player_pos, enemy_pos
   - Movement uses Manhattan neighbors (up, down, left, right), respecting
     grid boundaries.

2. Player controls and real-time loop
   - The program uses the curses library to capture non-blocking keyboard input.
   - Controls:
        w: move up
        s: move down
        a: move left
        d: move right
        p: pause / unpause the simulation
        q: quit
   - The game loop:
       - Polls for key presses (without blocking).
       - Updates player position immediately when WASD is pressed.
       - On a fixed time schedule (TICK_SECONDS), performs a simulation tick:
           * Update the enemy’s belief state.
           * Move the enemy according to its current best estimate of the
             player's position.
       - This separates input handling (continuous) from the logical simulation
         (discrete ticks).

3. Belief representation (Bayesian state)
   - The enemy does not know the player’s exact location (except when the
     player is visible). Instead, it maintains a probability distribution
     over all grid cells:
         belief[r][c] = P(player is at cell (r, c))
   - This 2D array is always normalized so that the sum over the grid is 1:
         sum_r sum_c belief[r][c] = 1
   - Initially, the belief is uniform:
         belief[r][c] = 1 / (HEIGHT * WIDTH)
     indicating maximum uncertainty.

4. Motion model (prediction step)
   - Function: motion_update(belief)
   - This implements the transition model P(x_t | x_{t-1}) in a Bayes filter.
   - Model assumption: from each cell, the player can:
        - stay in place
        - move N, S, E, or W (up to four neighbors)
     and all these options are equally likely, subject to grid boundaries.
   - For each cell (r, c) at time t-1 with probability belief[r][c]:
        * Enumerate its allowed destinations (including itself).
        * Distribute belief[r][c] equally among those destinations.
   - The resulting new_belief is the *predicted* distribution P(x_t) before
     any new observation at time t.

5. Observation models: vision and sound
   The enemy uses two sensor modalities:

   A. Vision (perfect, within radius)
      - If the true player position is within a specified Manhattan distance
        of the enemy (VISION_RADIUS), the enemy "sees" the player.
      - In this case, we treat the observation as exact and collapse the
        belief distribution to a point mass (delta) at the true player cell:
            belief[r][c] = 0 for all (r, c) != player_pos
            belief[player_row][player_col] = 1
      - After this update, the enemy moves to reduce its distance to the known
        player location (see "Enemy motion policy" below).

   B. Sound (noisy directional cue)
      - Sound is only generated when the player moves, and the enemy is within
        HEARING_RADIUS of the true player position.
      - Conceptually:
          * The *true* direction from enemy to player is bucketed into one of
            {N, S, E, W} by direction_bucket(enemy_pos, player_pos).
          * The enemy doesn't directly observe this true direction; instead
            it receives a noisy observation.
          * With probability P_CORRECT_DIR, the observed direction equals the
            true direction; with the remaining probability it is uniformly
            distributed among the other three directions.
      - The sound observation model is:

            P(observed_dir | player at cell x)
              = P_CORRECT_DIR        if direction_bucket(enemy, x) == observed_dir
              = (1 - P_CORRECT_DIR) / 3   otherwise

        (assuming four directions total and the observed_dir is one of them.)

      - Function: sound_update(belief, enemy_pos, observed_dir)
        implements the Bayesian correction step:

            Posterior(x) ∝ P(observed_dir | x) * Prior(x)

        where:
            - Prior(x) is the current belief(x) after motion_update.
            - P(observed_dir | x) is computed via the directional likelihood
              above, using direction_bucket(enemy_pos, x).
        After applying the likelihood to every cell, the distribution is
        renormalized so that it sums to 1.

6. Normalization
   - Function: normalize(belief)
   - Ensures the sum of probabilities is 1.
   - If numerical issues cause the total mass to be 0 (e.g., very unlikely
     observations), the belief is reset to a uniform distribution as a
     graceful fallback.

7. Direction bucketing and noisy sound generation
   - Function: direction_bucket(from_pos, to_pos)
        * Computes a coarse direction N/S/E/W from one cell to another.
        * It considers the dominant component of the vector (row vs column):
             - If |dr| >= |dc|, direction is North or South depending on the
               sign of dr.
             - Otherwise, direction is East or West depending on the sign of dc.
        * Returns None if the two positions are identical.

   - Function: sample_observed_dir(true_dir)
        * Takes the “true” coarse direction and returns a noisy observation,
          with probability P_CORRECT_DIR of being correct and otherwise a
          random choice among the remaining directions.

8. Enemy motion policy
   - Function: choose_enemy_move(enemy_pos, target_pos)
   - Given a target cell (enemy’s current best estimate of the player’s location),
     this function chooses a one-step move:
         - With small probability RANDOM_MOVE_PROB, the enemy takes a random
           valid step (to add some stochasticity).
         - Otherwise, it selects the neighbor that yields the smallest Manhattan
           distance to target_pos (a simple greedy hill-climbing step).
   - If enemy_pos == target_pos, the enemy stays in place.

9. Selecting the belief peak (most probable cell)
   - Function: find_belief_peak(belief)
   - Scans the entire grid, returning:
        - The cell (r, c) with maximum probability.
        - The corresponding probability value.
   - This peak cell is used as the "best guess" of the player’s location when
     the enemy does not see the player.

10. Game loop logic per tick
    For each simulation tick (when not paused and before the enemy has caught
    the player), the main sequence is:

        1. Capture player input continuously (outside of the tick timing):
             - Update player_pos immediately on WASD, constrained to the grid.

        2. On each tick (checked using wall-clock time and TICK_SECONDS):
             a. Determine whether the player moved since the previous tick.
                This is used to decide whether a sound event should be generated.

             b. Apply motion_update(belief) to perform the prediction step
                (player movement model).

             c. Check visibility:
                  - If manhattan(enemy_pos, player_pos) <= VISION_RADIUS:
                        * Collapse belief to a point mass at player_pos.
                        * Move enemy one step toward the known player position
                          (choose_enemy_move).

                  - Else (no direct vision):
                        * If the player moved AND is within HEARING_RADIUS:
                              - Compute the true direction from enemy to player.
                              - Sample a noisy observed direction.
                              - Apply sound_update(belief, enemy_pos, observed_dir)
                                to perform the Bayesian correction step using
                                the sound likelihood.
                          Otherwise:
                              - Just normalize(belief) to maintain a valid
                                distribution.

                        * Find the peak of the updated belief.
                        * Move the enemy one step toward this peak cell.

             d. Check for capture:
                  - If enemy_pos == player_pos, flag that the enemy has caught
                    the player.

        3. Between ticks, the loop sleeps briefly to avoid excessive CPU usage.

11. Visualization via curses
    - Function: draw_state(stdscr, player_pos, enemy_pos, belief, step, paused)
    - The terminal screen is refreshed each frame and shows:
        * A textual legend / status line at the top, including:
              step number, paused/run status, controls.
        * Left grid: the "actual world":
              P = player
              E = enemy
              X = both occupy the same cell
              . = empty cell
        * Right grid: the enemy’s belief as a simple ASCII heatmap:
              - A set of characters from light to dark, e.g. " .:-=+*#%@",
                is used to indicate relative probability density per cell.
              - The maximum-probability cell is marked with '@'.
        * A summary line showing:
              player_pos, enemy_pos, belief peak location, and its probability.
    - When the enemy catches the player, a message is displayed and the game
      waits for 'q' to quit.

Summary
-------
This program is an example of a simple online Bayesian filter ("hidden state
tracking") embedded in an interactive, real-time simulation:
    - The enemy maintains a belief distribution over the hidden state
      (player's location).
    - It predicts how that state evolves over time using a motion model.
    - It corrects that belief using noisy observations (directional sound)
      or a perfect observation (vision).
    - It then chooses actions (movements) based on its current belief,
      rather than having direct access to the true state, except when the
      player is visible.
"""

"""
Bayesian quantities in this program
===================================

The belief grid encodes the enemy’s uncertainty about the player's location.
For any cell x = (r, c):

    belief[r][c] = P(player is at cell x)

The Bayesian update each tick uses three components:

1. PRIOR
   -----
   The prior is the belief BEFORE the observation is incorporated.
   It comes from the motion model (prediction step) assuming how the player
   tends to move between ticks.

   - Initially: uniform distribution
         belief[r][c] = 1 / (HEIGHT * WIDTH)

   - After the first tick:
         prior_t(x) = sum over x' [ P(x | x') * posterior_{t-1}(x') ]
     where P(x | x') is the probability of moving from x' to x under the
     assumed player motion model (stay or move N/S/E/W with equal probability,
     respecting grid boundaries).

2. LIKELIHOOD
   -----------
   The likelihood expresses how compatible each cell x is with the new
   observation obtained by the enemy.

   Two kinds of observations exist:

   (a) Vision: perfect
       If the player is within VISION_RADIUS, the observation is exact.
       The likelihood is:
         P(observation | x) = 1 if x == true player position
                              0 otherwise
       Which collapses the belief to a delta distribution.

   (b) Sound: directional but noisy
       Let d_true(x) be the coarse direction from the enemy to cell x
       (N, S, E, or W). Let d_obs be the observed sound direction.
       The likelihood is:

         P(d_obs | x) =
             P_CORRECT_DIR                      if d_true(x) == d_obs
             (1 - P_CORRECT_DIR) / 3            if d_true(x) != d_obs but is defined
             0                                  if no defined direction to x

       Interpretation:
       Locations whose direction matches the observed sound are more likely
       to contain the player; others receive reduced probability.

3. POSTERIOR
   ---------
   The posterior is the corrected belief AFTER applying the observation.
   It follows Bayes’ rule:

       posterior(x) ∝ prior(x) * likelihood(x)

   In the code:
       new_belief[r][c] = belief[r][c] * likelihood

   This multiplies the prior probability for each cell by how consistent
   that cell is with what the enemy just sensed.

4. NORMALIZATION FACTOR
   ---------------------
   After multiplying by the likelihood, the grid no longer sums to 1.
   We compute:

       Z = sum_x [ prior(x) * likelihood(x) ]

   and divide every entry by Z:

       posterior(x) = (prior(x) * likelihood(x)) / Z

   Normalization ensures the belief remains a valid probability distribution.
   If Z = 0 (rare degenerate case), the system falls back to a uniform
   distribution to avoid numerical issues.

Summary of the Bayesian cycle each tick
---------------------------------------
    posterior_{t-1}
        → (motion_update)
    prior_t
        → (sound or vision likelihood)
    posterior_t
        → (enemy moves toward MAP estimate = argmax_x posterior_t(x))

Thus:
- PRIOR comes from the motion model.
- LIKELIHOOD comes from directional sound or perfect vision.
- POSTERIOR comes from multiplying PRIOR × LIKELIHOOD and normalizing.
- The NORMALIZATION FACTOR is the sum of the unnormalized posteriors.

The belief grid and this Bayesian cycle allow the enemy to reason about
player location even without direct sight, responding probabilistically
to noisy observations in real time.
"""

import curses
import random
import time
import math

# ----------------- CONFIGURATION -----------------

WIDTH = 10
HEIGHT = 10

HEARING_RADIUS = 5     # max distance at which enemy can hear player (Manhattan)
VISION_RADIUS = 4      # max distance at which enemy can see player (Manhattan)
P_CORRECT_DIR = 0.75   # likelihood sound direction is correct
RANDOM_MOVE_PROB = 0.1 # chance enemy moves randomly instead of towards target

TICK_SECONDS = 0.35     # delay between simulation steps

DIRECTIONS = {
    'w': (-1, 0),  # up
    's': (1, 0),   # down
    'a': (0, -1),  # left
    'd': (0, 1),   # right
}

# ----------------- UTILITY FUNCTIONS -----------------

# -----------------------------------------------------------------------------
# in_bounds(r, c)
# -----------------------------------------------------------------------------
# Purpose:
#   Check whether a grid coordinate (r, c) lies within the rectangular game
#   world of dimensions HEIGHT x WIDTH.
#
# Behavior:
#   - Returns True if 0 <= r < HEIGHT and 0 <= c < WIDTH.
#   - Returns False otherwise.
#
# Role in the model:
#   This function is a simple geometric constraint that ensures:
#     - Player and enemy do not move outside the grid.
#     - The transition model (motion_update) only allocates probability mass
#       to valid cells.
# -----------------------------------------------------------------------------
def in_bounds(r, c):
    return 0 <= r < HEIGHT and 0 <= c < WIDTH

# -----------------------------------------------------------------------------
# manhattan(p1, p2)
# -----------------------------------------------------------------------------
# Purpose:
#   Compute the Manhattan (L1) distance between two grid positions p1 and p2.
#
# Mathematical definition:
#   If p1 = (r1, c1) and p2 = (r2, c2), then
#       manhattan(p1, p2) = |r1 - r2| + |c1 - c2|.
#
# Behavior:
#   - Returns an integer >= 0.
#
# Role in the model:
#   - Used as a simple distance metric to:
#       * Define "hearing range" and "vision range" as distance thresholds.
#       * Decide how the enemy moves greedily toward a target cell in
#         choose_enemy_move (enemy reduces Manhattan distance each step).
# -----------------------------------------------------------------------------
def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# -----------------------------------------------------------------------------
# normalize(belief)
# -----------------------------------------------------------------------------
# Purpose:
#   Convert an arbitrary nonnegative belief grid into a valid probability
#   distribution that sums to 1.
#
# Mathematical effect:
#   Let belief[r][c] be nonnegative numbers and let
#       Z = sum_{r,c} belief[r][c].
#   If Z > 0:
#       normalized_belief[r][c] = belief[r][c] / Z.
#   If Z = 0 (degenerate case):
#       normalized_belief is reset to a uniform distribution:
#         1 / (HEIGHT * WIDTH) for all cells.
#
# Behavior:
#   - Always returns a 2D list of floats whose entries sum to 1.
#
# Role in the model:
#   This is the normalization step in Bayes' rule:
#       posterior(x) = (likelihood(x) * prior(x)) / Z
#   where Z is the normalizing constant ensuring probabilities sum to 1.
# -----------------------------------------------------------------------------
def normalize(belief):
    total = sum(sum(row) for row in belief)
    if total == 0:
        val = 1.0 / (WIDTH * HEIGHT)
        return [[val for _ in range(WIDTH)] for _ in range(HEIGHT)]
    return [[cell / total for cell in row] for row in belief]

# -----------------------------------------------------------------------------
# make_uniform_belief()
# -----------------------------------------------------------------------------
# Purpose:
#   Initialize a belief distribution that represents complete uncertainty
#   about the player's location: all grid cells are equally likely.
#
# Mathematical definition:
#   For all (r, c):
#       belief[r][c] = 1 / (HEIGHT * WIDTH).
#
# Behavior:
#   - Returns a HEIGHT x WIDTH grid of floats.
#   - Sum of all entries is exactly 1.
#
# Role in the model:
#   - Acts as an uninformative prior (maximum entropy) when the enemy has
#     no knowledge of the player's position.
#   - Also used as a fallback if normalization degenerates (see normalize).
# -----------------------------------------------------------------------------
def make_uniform_belief():
    val = 1.0 / (WIDTH * HEIGHT)
    return [[val for _ in range(WIDTH)] for _ in range(HEIGHT)]

# -----------------------------------------------------------------------------
# motion_update(belief)
# -----------------------------------------------------------------------------
# Purpose:
#   Perform the *prediction* step of a Bayes filter using a simple Markov
#   motion model for the player.
#
# Motion model:
#   From any cell x = (r, c), the player may:
#     - stay in place: x -> x
#     - move one step in any of the four cardinal directions:
#           x -> x + (±1, 0) or x -> x + (0, ±1)
#       provided the destination lies within the grid (in_bounds).
#
#   All allowed outcomes from a given cell are assumed *equally likely*.
#   If M(x) is the set of valid destinations from x (including x itself),
#   then:
#       P(X_t = y | X_{t-1} = x) = 1 / |M(x)|  if y ∈ M(x)
#                                = 0          otherwise
#
# Bayesian prediction:
#   The belief at time t, before incorporating any new observation, is:
#
#       belief_t(y) = sum over x [ P(X_t = y | X_{t-1} = x) * belief_{t-1}(x) ]
#
#   This is a discrete convolution with the transition kernel defined by the
#   motion model. The function implements this by:
#     1. Creating a new grid new_belief initialized to 0.
#     2. For each cell x, distributing belief[x] equally among its valid
#        destinations y ∈ M(x).
#     3. Returning new_belief as the predicted distribution.
#
# Behavior:
#   - Returns a HEIGHT x WIDTH grid of floats.
#   - The sum of these values will still be 1 (up to numerical error).
#
# Role in the model:
#   This function encodes how the enemy expects the player to move between
#   time steps in the absence of observations. It is the "time update" in
#   the Bayesian filtering process.
# -----------------------------------------------------------------------------
def motion_update(belief):
    """
    Prediction step: player can stay, or move N, S, E, W with equal probability
    from each cell (respecting boundaries).
    """
    new_belief = [[0.0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for r in range(HEIGHT):
        for c in range(WIDTH):
            moves = [(r, c)]
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc):
                    moves.append((nr, nc))
            p_move = belief[r][c] / len(moves)
            for nr, nc in moves:
                new_belief[nr][nc] += p_move
    return new_belief

# -----------------------------------------------------------------------------
# direction_bucket(from_pos, to_pos)
# -----------------------------------------------------------------------------
# Purpose:
#   Map the vector from 'from_pos' to 'to_pos' into a coarse cardinal
#   direction: North (N), South (S), East (E), or West (W).
#
# Geometric rule:
#   Let Δr = (to_row - from_row) and Δc = (to_col - from_col).
#   - If (Δr, Δc) = (0, 0), return None (no direction: same cell).
#   - Otherwise:
#       * If |Δr| >= |Δc|, use primarily vertical direction:
#             Δr > 0  => 'S'
#             Δr < 0  => 'N'
#       * Else, use primarily horizontal direction:
#             Δc > 0  => 'E'
#             Δc < 0  => 'W'
#
# Behavior:
#   - Returns one of {'N', 'S', 'E', 'W'} or None.
#
# Role in the model:
#   - Used to compute the "true" coarse direction from enemy to player,
#     which is then corrupted by noise to generate the observed sound
#     direction.
#   - Also used inside the likelihood computation for sound_update to
#     determine P(observed_dir | player at cell).
# -----------------------------------------------------------------------------
def direction_bucket(from_pos, to_pos):
    """
    Coarse direction from 'from_pos' to 'to_pos':
    returns 'N', 'S', 'E', 'W' or None if same cell.
    """
    fr, fc = from_pos
    tr, tc = to_pos
    dr = tr - fr
    dc = tc - fc
    if dr == 0 and dc == 0:
        return None
    if abs(dr) >= abs(dc):
        return 'S' if dr > 0 else 'N'
    else:
        return 'E' if dc > 0 else 'W'

# -----------------------------------------------------------------------------
# sample_observed_dir(true_dir)
# -----------------------------------------------------------------------------
# Purpose:
#   Generate a noisy observation of the sound direction given the true
#   coarse direction from enemy to player.
#
# Probabilistic model:
#   Let D = {'N', 'S', 'E', 'W'} be the set of directions.
#   Given true_dir ∈ D:
#     - With probability P_CORRECT_DIR, the observed direction equals true_dir.
#     - With probability 1 - P_CORRECT_DIR, the observed direction is drawn
#       uniformly from D \ {true_dir} (the other three directions).
#
#   If true_dir is None (degenerate case when positions coincide), we simply
#   choose uniformly from D.
#
# Behavior:
#   - Returns a single symbol in {'N', 'S', 'E', 'W'}.
#
# Role in the model:
#   - Implements the stochastic sensor model that converts a hidden true
#     direction into a noisy discrete observation used for Bayesian updating
#     in sound_update.
# -----------------------------------------------------------------------------
def sample_observed_dir(true_dir):
    """
    Sound direction observation with noise:
    with P_CORRECT_DIR it matches true_dir, otherwise random among the other 3.
    """
    dirs = ['N', 'S', 'E', 'W']
    if true_dir is None:
        return random.choice(dirs)
    if random.random() < P_CORRECT_DIR:
        return true_dir
    others = [d for d in dirs if d != true_dir]
    return random.choice(others)

# -----------------------------------------------------------------------------
# sound_update(belief, enemy_pos, observed_dir)
# -----------------------------------------------------------------------------
# Purpose:
#   Perform the *correction* step of a Bayes filter given a directional
#   sound observation.
#
# Observation model:
#   For any candidate player cell x, we define:
#       dir_to_x = direction_bucket(enemy_pos, x)
#
#   Let D = {'N', 'S', 'E', 'W'} with |D| = 4. We assume the following
#   conditional distribution:
#
#       P(observed_dir | X = x) =
#         P_CORRECT_DIR               if dir_to_x == observed_dir
#         (1 - P_CORRECT_DIR) / (|D|-1)   if dir_to_x ∈ D and dir_to_x != observed_dir
#         0                           if dir_to_x is None (enemy and x coincide)
#
#   This expresses that the observed direction is likely to match the true
#   coarse direction but may be any of the others with lower probability.
#
# Bayesian update:
#   Let prior(x) = belief[x] be the predicted belief before seeing the
#   sound observation. The posterior after observing observed_dir is
#
#       posterior(x) ∝ P(observed_dir | X=x) * prior(x)
#
#   In detail:
#       new_belief[x] = prior(x) * likelihood(x)
#     where:
#       likelihood(x) = P(observed_dir | X=x)
#
#   After computing new_belief for every cell x, we call normalize(new_belief)
#   so that the probabilities sum to 1.
#
# Behavior:
#   - If observed_dir is None, returns the input belief unchanged.
#   - Otherwise returns a new normalized belief grid reflecting the sound
#     information.
#
# Role in the model:
#   This function injects information from noisy sound into the enemy's
#   belief about the player's location, shrinking probability mass toward
#   cells whose geometric direction from the enemy matches the observed
#   direction more closely.
# -----------------------------------------------------------------------------
def sound_update(belief, enemy_pos, observed_dir):
    """
    Bayesian update P(x | observed_dir) = P(observed_dir | x) * P(x)
    where x is a candidate player cell.
    """
    if observed_dir is None:
        return belief
    dirs = ['N', 'S', 'E', 'W']
    n_dirs = len(dirs)
    p_correct = P_CORRECT_DIR
    p_incorrect = (1.0 - P_CORRECT_DIR) / (n_dirs - 1)

    new_belief = [[0.0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for r in range(HEIGHT):
        for c in range(WIDTH):
            dir_to_cell = direction_bucket(enemy_pos, (r, c))
            if dir_to_cell is None:
                likelihood = 0.0
            elif dir_to_cell == observed_dir:
                likelihood = p_correct
            else:
                likelihood = p_incorrect
            new_belief[r][c] = belief[r][c] * likelihood

    return normalize(new_belief)

# -----------------------------------------------------------------------------
# choose_enemy_move(enemy_pos, target_pos)
# -----------------------------------------------------------------------------
# Purpose:
#   Choose a single-step move for the enemy, attempting to reduce its
#   Manhattan distance to the target cell, with some randomness.
#
# Behavior:
#   1. If enemy_pos == target_pos, stay in place.
#   2. With small probability RANDOM_MOVE_PROB:
#        - Move to a uniformly random neighboring cell (up/down/left/right)
#          that lies within the grid.
#   3. Otherwise (the usual case):
#        - Among all valid neighboring cells (including up/down/left/right),
#          choose the one that minimizes the Manhattan distance to target_pos.
#        - If multiple neighbors tie, the first encountered is chosen.
#
# Mathematical effect:
#   - Step 3 implements a greedy descent in the Manhattan distance metric:
#         d_new = min_{neighbors n} manhattan(n, target_pos)
#         with d_new <= d_old
#   - Step 2 introduces stochasticity so that the enemy's path is not
#     entirely deterministic and can occasionally deviate from the
#     mathematically optimal path.
#
# Role in the model:
#   - When the enemy has *perfect* knowledge (vision), target_pos is the
#     actual player location.
#   - When relying on the belief, target_pos is chosen as the maximum
#     a posteriori (MAP) estimate (the belief peak). The enemy then moves
#     as if that were the true state.
# -----------------------------------------------------------------------------
def choose_enemy_move(enemy_pos, target_pos):
    """
    One-step enemy move toward target (with occasional random step).
    """
    er, ec = enemy_pos
    if enemy_pos == target_pos:
        return enemy_pos

    if random.random() < RANDOM_MOVE_PROB:
        candidates = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = er + dr, ec + dc
            if in_bounds(nr, nc):
                candidates.append((nr, nc))
        return random.choice(candidates) if candidates else enemy_pos

    tr, tc = target_pos
    best = enemy_pos
    best_dist = manhattan(enemy_pos, target_pos)
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = er + dr, ec + dc
        if not in_bounds(nr, nc):
            continue
        d = manhattan((nr, nc), target_pos)
        if d < best_dist:
            best_dist = d
            best = (nr, nc)
    return best

# ----------------- VISUALIZATION -----------------

# -----------------------------------------------------------------------------
# find_belief_peak(belief)
# -----------------------------------------------------------------------------
# Purpose:
#   Identify the grid cell with the highest posterior probability
#   under the current belief distribution.
#
# Mathematical definition:
#   Given belief[r][c] for all (r, c), we compute:
#
#       (r*, c*) = argmax_{r,c} belief[r][c]
#       p*       = max_{r,c}   belief[r][c]
#
#   The function returns both the maximizing cell (r*, c*) and value p*.
#
# Behavior:
#   - Scans all cells, tracking the maximum probability and its coordinates.
#   - Returns (peak_cell, peak_prob) where:
#       peak_cell is a (row, col) tuple.
#       peak_prob is the corresponding float probability.
#
# Role in the model:
#   The peak cell is used as the enemy's *best guess* of the player's
#   location when the player is not directly visible. The enemy then uses
#   this cell as the target for choose_enemy_move, effectively implementing
#   MAP-based pursuit:
#
#       target = argmax_x belief(x)
# -----------------------------------------------------------------------------
def find_belief_peak(belief):
    max_prob = -1.0
    peak = (0, 0)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if belief[r][c] > max_prob:
                max_prob = belief[r][c]
                peak = (r, c)
    return peak, max_prob

# -----------------------------------------------------------------------------
# draw_state(stdscr, player_pos, enemy_pos, belief, step, paused)
# -----------------------------------------------------------------------------
# Purpose:
#   Render the current simulation state in the terminal using curses.
#
# Visual layout:
#   - Line 0: Status line with:
#       * Current time step (step)
#       * Pause state
#       * Key hints (q=quit, p=pause)
#
#   - Left grid (starting around row 2):
#       "Actual world" view:
#           P : player position
#           E : enemy position
#           X : both on the same cell
#           . : empty cell
#
#   - Right grid:
#       Belief heatmap:
#         * For each cell, map its probability to an ASCII character chosen
#           from a gradient string such as " .:-=+*#%@" where darker symbols
#           represent higher probabilities.
#         * The maximum-probability cell is highlighted as '@'.
#
#   - Bottom lines:
#       * Numeric summary:
#             Player position, enemy position,
#             location of belief peak, and its probability.
#       * Additional help text describing controls and behavior.
#
# Scaling of the heatmap:
#   - Let p_max = max_{r,c} belief[r][c].
#   - Each cell's relative intensity is:
#         level = int((belief[r][c] / p_max) * (len(shades) - 1))
#     clipped to the range [0, len(shades)-1].
#
# Role in the model:
#   This function does not change the state; it only visualizes:
#     - The "ground truth" (actual positions).
#     - The enemy's internal probabilistic belief.
#   This helps to interpret how the Bayesian updates evolve over time.
# -----------------------------------------------------------------------------
def draw_state(stdscr, player_pos, enemy_pos, belief, step, paused):
    stdscr.clear()

    peak_cell, peak_prob = find_belief_peak(belief)

    # Title / status
    status_line = f"Step: {step}   PAUSED: {'YES' if paused else 'NO'}   q=quit, p=pause"
    stdscr.addstr(0, 0, status_line)

    # Actual grid (left)
    start_row = 2
    start_col_left = 0
    stdscr.addstr(start_row - 1, start_col_left, "Actual world (P=player, E=enemy, X=both):")
    for r in range(HEIGHT):
        row_chars = []
        for c in range(WIDTH):
            if (r, c) == player_pos and (r, c) == enemy_pos:
                ch = 'X'
            elif (r, c) == player_pos:
                ch = 'P'
            elif (r, c) == enemy_pos:
                ch = 'E'
            else:
                ch = '.'
            row_chars.append(ch)
        stdscr.addstr(start_row + r, start_col_left, " ".join(row_chars))

    # Belief grid (right): heatmap-like using ASCII shades
    shades = " .:-=+*#%@"  # low -> high
    start_col_right = 3 + 2 * WIDTH
    stdscr.addstr(start_row - 1, start_col_right,
                  "Enemy belief (heatmap, @ = highest):")

    # For scaling, use the maximum probability value
    peak_r, peak_c = peak_cell
    max_prob = peak_prob if peak_prob > 0 else 1e-9

    for r in range(HEIGHT):
        row_chars = []
        for c in range(WIDTH):
            p = belief[r][c]
            # Scale probability relative to peak
            level = int((p / max_prob) * (len(shades) - 1))
            level = max(0, min(level, len(shades) - 1))
            ch = shades[level]

            # Mark the peak explicitly
            if (r, c) == peak_cell:
                ch = '@'
            row_chars.append(ch)
        stdscr.addstr(start_row + r, start_col_right, " ".join(row_chars))

    # Info lines at bottom
    bottom_row = start_row + HEIGHT + 1
    stdscr.addstr(bottom_row, 0,
                  f"Player at {player_pos}, Enemy at {enemy_pos}, "
                  f"Belief peak at {peak_cell} (prob ≈ {peak_prob:.3f})")
    stdscr.addstr(bottom_row + 1, 0,
                  "Use WASD to move. Enemy moves each tick based on belief or vision.")
    stdscr.addstr(bottom_row + 2, 0,
                  "If enemy sees you (within vision radius), it chases directly.")

    stdscr.refresh()

# ----------------- MAIN LOOP (CURSES) -----------------

# -----------------------------------------------------------------------------
# game_loop(stdscr)
# -----------------------------------------------------------------------------
# Purpose:
#   Main real-time control loop for the simulation, wrapped by curses.
#
# Responsibilities:
#   1. Initialize game state:
#        - Random player and enemy starting positions (distinct).
#        - Initial uniform belief over player positions.
#        - Time step counter 'step', pause flag 'paused', catch flag 'caught'.
#
#   2. Configure curses:
#        - Hide cursor.
#        - Make getch() non-blocking (nodelay + timeout).
#
#   3. Continuous loop:
#        - Record current wall-clock time.
#        - Draw the current state using draw_state().
#
#        - Display message depending on whether the enemy has caught the player.
#
#        - Handle user input:
#            q : quit the simulation.
#            p : toggle pause/unpause.
#          WASD : move the player (if not caught) by a single step in bounds.
#          (Player movement is applied immediately as keys are pressed.)
#
#        - Simulation tick timing:
#            * A tick occurs whenever (current_time - last_time) >= TICK_SECONDS
#              and the game is not paused and not yet caught.
#
#            On each tick:
#              a. Determine whether the player moved since the previous tick
#                 by comparing player_pos and a stored prev_player_pos.
#              b. Apply motion_update(belief) to predict where the player
#                 might be after movement.
#              c. Check direct visibility via manhattan(enemy_pos, player_pos):
#                   - If within VISION_RADIUS:
#                         Collapse belief to a delta at the true player_pos,
#                         then move enemy one step toward player_pos.
#                   - Else (no direct sight):
#                         If player moved and is within HEARING_RADIUS:
#                             • Compute true direction (enemy -> player).
#                             • Sample noisy observed direction.
#                             • Apply sound_update() for Bayesian correction.
#                         Otherwise:
#                             • Just normalize() to keep a valid distribution.
#
#                         Then:
#                             • Find belief peak (MAP estimate).
#                             • Move enemy one step toward that peak via
#                               choose_enemy_move().
#
#              d. After moving the enemy, check for capture:
#                    if enemy_pos == player_pos: caught = True.
#
#        - Between loop iterations, sleep briefly (time.sleep) to limit CPU
#          usage while still allowing responsive real-time input.
#
# Role in the model:
#   This function orchestrates:
#     - The Bayesian filtering cycle (motion_update + sound/vision updates).
#     - The enemy's decision-making based on beliefs.
#     - The rendering of internal and external state.
#   It effectively couples continuous input, discrete-time probabilistic
#   inference, and action selection in a single real-time loop.
# -----------------------------------------------------------------------------
def game_loop(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)   # non-blocking getch
    stdscr.timeout(0)      # return immediately on getch

    random.seed()

    # Random starting positions
    player_pos = (random.randrange(HEIGHT), random.randrange(WIDTH))
    while True:
        enemy_pos = (random.randrange(HEIGHT), random.randrange(WIDTH))
        if enemy_pos != player_pos:
            break

    belief = make_uniform_belief()
    step = 0
    paused = False
    caught = False

    last_time = time.time()

    while True:
        now = time.time()
        # Draw current state
        draw_state(stdscr, player_pos, enemy_pos, belief, step, paused)

        msg_row = HEIGHT + 5   # choose the row you want your message on

        # Clear that entire line first
        stdscr.move(msg_row, 0)
        stdscr.clrtoeol()

        if caught:
            stdscr.addstr(msg_row, 0, "The enemy has caught the player! Press q to quit.")
        else:
            stdscr.addstr(msg_row, 0, "Press p to pause/unpause. q to quit.")

        # Handle input
        ch = stdscr.getch()
        if ch != -1:
            try:
                key = chr(ch).lower()
            except ValueError:
                key = ''
            if key == 'q':
                break
            elif key == 'p':
                paused = not paused
            elif key in DIRECTIONS and not caught:
                # Move player immediately on keypress
                dr, dc = DIRECTIONS[key]
                nr = player_pos[0] + dr
                nc = player_pos[1] + dc
                if in_bounds(nr, nc):
                    player_pos = (nr, nc)

        # Check if enough time has passed for a simulation tick
        if not paused and not caught and (now - last_time) >= TICK_SECONDS:
            last_time = now
            step += 1

            # At each tick, we consider whether player moved during this tick:
            # since we update immediately on keypress, the movement is already
            # reflected in player_pos. For sound, we approximate that the player
            # "moved" (and thus made sound) if their position changed since the
            # last tick. So we store previous position before belief update.
            # To do this, we need to keep previous_player_pos across ticks.
            # We'll implement that by storing it as an attribute on the function
            # object or using closure; here we use a global-like variable.

            if not hasattr(game_loop, "prev_player_pos"):
                game_loop.prev_player_pos = player_pos

            old_player_pos = game_loop.prev_player_pos
            player_moved = (player_pos != old_player_pos)
            game_loop.prev_player_pos = player_pos

            # ENEMY BELIEF PREDICTION
            belief = motion_update(belief)

            # Direct vision check
            visible = manhattan(enemy_pos, player_pos) <= VISION_RADIUS

            if visible:
                # Collapse belief to exact player position
                belief = [[0.0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
                pr, pc = player_pos
                belief[pr][pc] = 1.0
                enemy_pos = choose_enemy_move(enemy_pos, player_pos)
            else:
                # Sound update if close enough and player moved
                if player_moved and manhattan(enemy_pos, player_pos) <= HEARING_RADIUS:
                    true_dir = direction_bucket(enemy_pos, player_pos)
                    observed_dir = sample_observed_dir(true_dir)
                    belief = sound_update(belief, enemy_pos, observed_dir)
                else:
                    belief = normalize(belief)

                # Move enemy towards most probable cell
                peak_cell, _ = find_belief_peak(belief)
                enemy_pos = choose_enemy_move(enemy_pos, peak_cell)

            # Check capture
            if enemy_pos == player_pos:
                caught = True

        # Small sleep to avoid busy-looping; interaction is still real-time.
        time.sleep(0.01)

def main():
    curses.wrapper(game_loop)

if __name__ == "__main__":
    main()

