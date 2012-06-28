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
import System.Random.Shuffle
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
data GeneticSettings d b = GeneticSettings {
    maxIterations :: Int -- ^ The maximum number of iterations to run before giving up. 
  , targetFitness :: Double -- ^ The simulation will stop once the required portion of the population reaches this level.
  , targetPercent :: Double -- ^ The amount of the population that must reach the target fitness.
  , maxParents :: Int -- ^ The number of parents that are used to generate offspring
  , breeding :: (StdGen,[[b]]) -> (StdGen,[[b]]) -- ^ The breeding function that should be used.
}

-- | sane defaults
defaultSettings :: (DNA d) => GeneticSettings d b
defaultSettings = GeneticSettings {
    maxIterations = 100
  , targetFitness = 0.9
  , targetPercent = 0.9
  , maxParents = 2
  , breeding = crossover
}


-- | Runs the simulation and returns the best solution.
evolve :: (DNA d) => StdGen -> GeneticSettings d b -> ([d] -> [(d,Fitness)]) -> [d] -> d
evolve gen settings fitness initialPopulation = best
  where
    populationSize = length initialPopulation
    finalPopulation = run 0 gen (fitness initialPopulation)
    (best,_) = maximumBy (\(_,a) (_,b) -> compare a b) finalPopulation

    --step :: StdGen -> [(d,Fitness)] -> (StdGen,[(d,Fitness)])
    step gen population = (gen',fitness population')
      where
        (gen',population') = breed gen population

    --breed :: (DNA d) => StdGen -> [(d,Fitness)] -> (StdGen,[d])
    breed gen population = (gen',population')
      where
        encodedPopulation = map (\(d,f) -> (encode d,f)) population
        (gen',infEncodedPopulationList) = mapAccumL (\g b -> b (getParents g)) gen $ repeat (breeding settings)
        population' = map decode $ take populationSize $ concat infEncodedPopulationList 

        --getParents :: StdGen -> (StdGen,[[b]])
        getParents g = (g', take (maxParents settings) parents)
          where
            ((g',_), parents) = mapAccumL (\acc pp -> pp acc) (g,encodedPopulation) $ replicate populationSize pickParent

            --pickParent :: (StdGen,[[b]]) -> ((StdGen,[[b]]),[b])
            pickParent (g,population) = ((g',population'), parent)
              where
                (fitnessSum, populationDistribution) = mapAccumL (\s (d,f) -> (f+s,(d,f+s))) 0 population
                (r,g') = randomR (0,fitnessSum) g
                --split the population distribution at r
                (popLessThanR, popMoreThanR) = span (\(d,f) -> f<r) populationDistribution
                --the chosen parent is the head of the list that is greater than r
                (parent,_) = head popMoreThanR
                population' = popLessThanR ++ (tail popMoreThanR)

    --run :: Int -> StdGen -> [d] -> [(d,Fitness)]
    run iteration gen population
        | iteration+1 >= (maxIterations settings)       = population' -- stop at max iterations
        | percentOverTarget >= (targetPercent settings) = population' -- stop at found fitness
        | otherwise                                     = run (iteration+1) gen' $ population' --keep going
          where
            (gen',population') = step gen population -- next population
            numOverTarget = fromIntegral $ length $ filter (\(_,f) -> f >= (targetFitness settings)) population'
            percentOverTarget = numOverTarget / (fromIntegral $ length population')


-- | Performs simple crossover breeding. N-1 pivots in the DNA are chosen, where N is the number of parents. Children are created by all combinations of resulting parent segments.
crossover :: (StdGen, [[b]]) -> (StdGen, [[b]])
crossover (gen,parents) = (gen',children)
  where
    (gen',dividedParents) = splitParents gen (length parents - 1) parents
    children = combine dividedParents

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
