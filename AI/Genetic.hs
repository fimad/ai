module AI.Genetic (
  -- * Basic Usage
  -- * Classes and Types
    DNA (..)
  , Fitness
  -- ** Simulation Settings
  , GeneticSettings (..)
  , defaultSettings
  -- * Simulations
  , evolve
  -- * Breeding Functions
  , crossover
)
where

import Data.List
import System.Random
import Data.Word as W

-- | A typeclass that represents a solution that can be found using a genetic algorithm.
class DNA a where
  -- | encodes a solution a as a list of doubles.
  encode :: a -> [b]
  -- | takes an genetic encoding of a solution and returns the corresponding solution.
  decode :: [b] -> a

-- | Fitness is a positive Double. Larger values correspond with better fitness.
type Fitness = Double


-- | Settings for how a simulation should be run.
data GeneticSettings d = GeneticSettings {
    maxIterations :: Int -- ^ The maximum number of iterations to run before giving up. 
  , targetFitness :: Double -- ^ The simulation will stop once the required portion of the population reaches this level.
  , targetPercent :: Double -- ^ The amount of the population that must reach the target fitness.
  , parents :: Int -- ^ The number of parents that are used to generate offspring
  , fitness :: [d] -> [(d,Fitness)] -- ^ The fitness function.
  , breeding :: StdGen -> [d] -> (StdGen,[d]) -- ^ The breeding function that should be used.
}

-- | sane defaults, a fitness function must still be supplied because the provided one assigns a Fitness of 0 to all candidates.
defaultSettings :: (DNA d) => GeneticSettings d
defaultSettings = GeneticSettings {
    maxIterations = 100
  , targetFitness = 0.9
  , targetPercent = 0.9
  , parents = 2
  , breeding = crossover
  , fitness = map (\x -> (x,0)) -- a dummy fitness function, should never actually be used
}


-- | Runs the simulation and returns the best solution.
evolve :: (DNA d) => StdGen -> GeneticSettings d -> [d] -> d
evolve gen settings initialPopulation = best
  where
    populationSize = length initialPopulation
    finalPopulation = run 0 gen initialPopulation
    (best,_) = maximumBy (\(_,a) (_,b) -> compare a b) finalPopulation

    --step :: (DNA d) => StdGen -> [d] -> (StdGen,[(d,Fitness)])
    step gen population = (gen,(fitness settings) population)

    --breed :: (DNA d) => StdGen -> [d] -> (StdGen,[d])
    breed gen population = (gen,population)

    --run :: Int -> StdGen -> [d] -> [(d,Fitness)]
    run iteration gen population
        | iteration+1 >= (maxIterations settings)       = population' -- stop at max iterations
        | percentOverTarget >= (targetPercent settings) = population' -- stop at found fitness
        | otherwise                                     = run (iteration+1) gen' $ map fst population' --keep going
          where
            (gen',population') = step gen population -- next population
            numOverTarget = fromIntegral $ length $ filter (\(_,f) -> f >= (targetFitness settings)) population'
            percentOverTarget = numOverTarget / (fromIntegral $ length population')


-- | Performs simple crossover breeding. N-1 pivots in the DNA are chosen, where N is the number of parents. Children are created by all combinations of resulting parent segments.
crossover :: (DNA d) => StdGen -> [d] -> (StdGen, [d])
crossover gen parents = (gen',children)
  where
    encodedParents = map encode parents
    (gen',dividedParents) = splitParents gen (length parents - 1) encodedParents
    encodedChildren = combine dividedParents
    children = map decode encodedChildren

    -- Creates pivot points and splits parents along the pivots.
    splitParents :: StdGen -> Int -> [[b]] -> (StdGen,[[[b]]])
    splitParents gen 0 _ = (gen,[])
    splitParents gen i parents = (gen'', heads : tails)
      where
        (r,gen') = next gen
        pivot = r `mod` ((length $ head parents) - (length parents))
        heads = map (take pivot) parents
        (gen'',tails) = splitParents gen' (i-1) $ map (drop pivot) parents

    -- Combinatorially combines the parent segments to create the offspring.
    combine :: [[[b]]] -> [[b]]
    combine [] = []
    combine (x:xs) = x >>= (\x -> map (x++) $ combine xs)
