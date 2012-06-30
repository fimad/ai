module AI.Genetic (
  -- | A library that implements a genetic search algorithm.

    Fitness
  , GeneticSettings (..)
  , defaultSettings
  , evolve
  , randomPopulation
  , fitnessMap
)
where

import Data.List
import Control.Monad
import System.Random
import System.Random.Shuffle
import Data.Word as W
import Data.Tuple as T
import AI.Genetic.Breeding
import AI.Genetic.Selection

-- | Fitness is a positive Double. Larger values correspond with better fitness.
type Fitness = Double

-- | Settings for how a simulation should be run.
data GeneticSettings b = GeneticSettings {
    maxIterations :: Int -- ^ The maximum number of iterations to run before giving up. 
  , targetFitness :: Double -- ^ The simulation will stop once the required portion of the population reaches this level.
  , targetPercent :: Double -- ^ The amount of the population that must reach the target fitness.
  , maxParents :: Int -- ^ The number of parents that are used to generate offspring
  , maxChildren :: Int -- ^ The maximum number of children that is allowed to result from one set of parents breeding.
  , validDNA :: [b] -> Bool -- ^ A predicate for filtering out invalid solutions during breeding.
  , breeding :: (StdGen,[[b]]) -> (StdGen,[[b]]) -- ^ The breeding function that should be used. See "AI.Genetic.Breeding" for possible genetic operations.
  , selection :: (StdGen,[([b],Fitness)]) -> ((StdGen,[([b],Fitness)]),[b]) -- ^ The selection function determines how parents are chosen from the population for breeding. See "AI.Genetic.Selection" for options.
}

-- | Reasonable defaults that can be passed to 'evolve'.
defaultSettings :: (Random b, Eq b) => GeneticSettings b
defaultSettings = GeneticSettings {
    maxIterations = 100
  , targetFitness = 1
  , targetPercent = 0.00000000001
  , maxParents = 2
  , maxChildren = 2
  , validDNA = (\_ -> True)
  , breeding = crossover . (mutation 0.001)
  , selection = rouletteWheel
}

-- | Generates a random population, the solution type must be a member of the Random class.
randomPopulation :: (Random b, Integral i)
  => StdGen
  -> GeneticSettings b -- ^ a genetic settings needs to be passed in so that the validDNA function maybe used.
  -> i -- ^ The length of the dna to generate.
  -> i -- ^ The size of the population.
  -> [[b]]
randomPopulation gen settings dnaSize populationSize =
    take (fromIntegral populationSize)
  $ filter (validDNA settings)
  $ snd
  $ mapAccumL (\gen' rs -> rs gen') gen
  $ repeat randomSolution
  where
    randomSolution gen = mapAccumL (\gen' r -> swap $ r gen') gen $ replicate (fromIntegral dnaSize) random


-- | Takes a fitness function that operates on a single solution and returns a fitness function suitable for use with 'evolve'.
fitnessMap ::
  ([b] -> Fitness) -- ^ A function that returns the fitness for a single solution
  -> ([[b]] -> [([b],Fitness)]) -- ^ A fitness function suitable for use with this library.
fitnessMap f = map (\x -> (x,f x))


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

    --breed :: StdGen -> [(d,Fitness)] -> (StdGen,[d])
    breed gen population = (gen',population')
      where
        (gen',infEncodedPopulationList) = mapAccumL (\g b -> b (getParents g)) gen $ replicate populationSize (pickChildren . breeding settings)
        population' = take populationSize $ concat infEncodedPopulationList 
        --getParents :: StdGen -> (StdGen,[[b]])
        getParents g = (g', take (maxParents settings) parents)
          where
            ((g',_), parents) = mapAccumL (\acc pp -> pp acc) (g,population) $ replicate (maxParents settings) (selection settings)

        --filters out bad children and randomly discards down to maxChildren if needed
        pickChildren (gen,children) = (gen,take (maxChildren settings) randomChildren)
          where
            validChildren = filter (validDNA settings) children
            randomChildren = if null validChildren then [] else shuffle' validChildren (length validChildren) gen

    --run :: Int -> StdGen -> [d] -> [(d,Fitness)]
    run iteration gen population
        | iteration+1 >= (maxIterations settings)       = population' -- stop at max iterations
        | percentOverTarget >= (targetPercent settings) = population' -- stop at found fitness
        | otherwise                                     = run (iteration+1) gen' $ population' --keep going
          where
            (gen',population') = step gen population -- next population
            numOverTarget = fromIntegral $ length $ filter (\(_,f) -> f >= (targetFitness settings)) population'
            percentOverTarget = numOverTarget / (fromIntegral $ length population')
