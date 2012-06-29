module AI.Genetic (
  -- | A library that implements a genetic search algorithm.

  -- * Classes and Types
    Fitness
  -- ** Simulation Settings
  , GeneticSettings (..)
  , defaultSettings
  -- * Simulations
  , evolve
  , randomPopulation
  -- * Breeding Functions
  -- | Each breeding function implements a simple genetic operation, and may be composed with others to create more powerful operations.
  , crossover
  , mutation
)
where

import Data.List
import Control.Monad
import System.Random
import System.Random.Shuffle
import Data.Word as W
import Data.Tuple as T

-- | Fitness is a positive Double. Larger values correspond with better fitness.
type Fitness = Double

-- | Settings for how a simulation should be run.
data GeneticSettings b = GeneticSettings {
    maxIterations :: Int -- ^ The maximum number of iterations to run before giving up. 
  , targetFitness :: Double -- ^ The simulation will stop once the required portion of the population reaches this level.
  , targetPercent :: Double -- ^ The amount of the population that must reach the target fitness.
  , maxParents :: Int -- ^ The number of parents that are used to generate offspring
  , breeding :: (StdGen,[[b]]) -> (StdGen,[[b]]) -- ^ The breeding function that should be used.
}

-- | sane defaults
defaultSettings :: (Random b, Eq b) => GeneticSettings b
defaultSettings = GeneticSettings {
    maxIterations = 100
  , targetFitness = 1
  , targetPercent = 0.00000000001
  , maxParents = 2
  , breeding = crossover . (mutation 0.001)
}

-- | Generates a random population, the solution type must be a member of the Random class.
randomPopulation :: (Random b, Integral i) => StdGen -> i -> i -> [[b]]
randomPopulation gen dnaSize populationSize = snd $ mapAccumL (\gen' rs -> rs gen') gen $ replicate (fromIntegral populationSize) randomSolution
  where
    randomSolution gen = mapAccumL (\gen' r -> swap $ r gen') gen $ replicate (fromIntegral dnaSize) random

-- | Runs the simulation and returns the best solution.
evolve :: (Eq b) => StdGen
  -> GeneticSettings b -- ^ Settings for running the simulation, 'defaultSettings' are sane defaults.
  -> ([[b]] -> [([b],Fitness)]) -- ^ The fitness function for ranking potential solutions.
  -> [[b]] -- ^ The initial population.
  -> [b] -- ^ Resulting best solution found by the simulation.
evolve gen settings fitness initialPopulation = best
  where
    populationSize = length initialPopulation
    finalPopulation = run 0 gen (fitness initialPopulation)
    (best,_) = maximumBy (\(_,a) (_,b) -> compare a b) finalPopulation

    --step :: StdGen -> [(d,Fitness)] -> (StdGen,[(d,Fitness)])
    step gen population = (gen',fitness population')
      where
        (gen',population') = breed gen population

    --breed :: ([b] d) => StdGen -> [(d,Fitness)] -> (StdGen,[d])
    breed gen population = (gen',population')
      where
        (gen',infEncodedPopulationList) = mapAccumL (\g b -> b (getParents g)) gen $ replicate populationSize (breeding settings)
        population' = take populationSize $ concat infEncodedPopulationList 

        --getParents :: StdGen -> (StdGen,[[b]])
        getParents g = (g', take (maxParents settings) parents)
          where
            ((g',_), parents) = mapAccumL (\acc pp -> pp acc) (g,population) $ replicate (maxParents settings) pickParent

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


-- | Performs simple crossover breeding. N-1 pivots in the [b] are chosen, where N is the number of parents. Children are created by all combinations of resulting parent segments.
crossover :: (Eq b) => (StdGen, [[b]]) -> (StdGen, [[b]])
crossover (gen,parents) = (gen',children)
  where
    (gen',dividedParents) = splitParents gen (length parents - 1) parents
    children = combine dividedParents

    -- Creates pivot points and splits parents along the pivots.
    --splitParents :: StdGen -> Int -> [[b]] -> (StdGen,[[[b]]])
    splitParents gen 0 parents = (gen,[parents])
    splitParents gen i parents = (gen'', heads : tails)
      where
        (r,gen') = next gen
        pivot = r `mod` ((length $ head parents) - (length parents)+1)+1
        heads = map (take pivot) parents
        (gen'',tails) = splitParents gen' (i-1) $ map (drop pivot) parents

    -- Combinatorially combines the parent segments to create the offspring.
    --combine :: [[[b]]] -> [[b]]
    combine = filter (\x -> not $ elem x parents) . foldM (\acc x -> map (acc++) x) []


-- | Randomly mutates genes at a given frequency.
mutation :: (Random b)
  => Double -- ^ Frequency at which genes are mutated, must be between 0 and 1. By partially applying the frequency, the result is a composable genetic operation.
  -> (StdGen, [[b]])
  -> (StdGen, [[b]])
mutation frequency (gen,parents) = mapAccumL (\gen' bs -> mapAccumL mutateGene gen' bs) gen parents
  where
    --mutateGene :: StdGen -> b -> (StdGen,b)
    mutateGene gen b = if r <= frequency then T.swap $ random gen' else (gen,b)
      where
        (r,gen') = randomR (0.0,1.0) gen

