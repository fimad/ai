module AI.Genetic (
  -- * How To Use
  -- | There are two prerequisites to training something using this library.

  -- | The first is to choose a genetic encoding for potential solutions. Genetic encodings are lists of genes, if your problem space is already represented this way then you are done, if not you will need to encode/decode all solutions coming in to/out of this library.

  -- | The second is to write a 'FitnessFunction'. Fitness functions assign a 'Fitness' to each member of a population. This value can be based either on the performance of an individual to the group or solely based on an individual. The only requirements is that it is >= 0, with larger values being \"more fit\". It is also worthwhile to keep in mind that for selection algorithms, individuals will be chosen proportionally to their fitness.

  -- ** Training
    evolveI
  , evolveP
  , evolveStep
  , evolveStep'
  , randomPopulation

  -- *** Fine Tuning
  -- | There are various settings that control the progression of the genetic search, and they can be set by supplying a 'GeneticSettings' value to the 'evolve' functions. To quickly get up and running, 'defaultSettings' contains sane default vallues for each value.

  , GeneticSettings (..)
  , defaultSettings

  -- ** Fitness Function

  , Fitness
  , FitnessFunction
  , fitnessMap
)
where

import Data.List
import Control.Monad
import System.Random
import Data.Tuple as T
import AI.Genetic.Breeding
import AI.Genetic.Selection
import System.Random.Shuffle

-- | Fitness is a positive Double. Larger values correspond with better fitness.
type Fitness = Double

-- | The fitness function scores assigns a 'Fitness' value to each member of a population.
type FitnessFunction b = ([[b]] -> [([b],Fitness)])

-- | Settings for how a simulation should be run.
data GeneticSettings b = GeneticSettings {
    maxIterations :: Int -- ^ The maximum number of iterations to run before giving up. 
  , targetFitness :: Double -- ^ The simulation will stop once the required portion of the population reaches this level.
  , targetPercent :: Double -- ^ The amount of the population that must reach the target fitness.
  , maxParents :: Int -- ^ The number of parents that are used to generate offspring
  , maxChildren :: Int -- ^ The maximum number of children that is allowed to result from one set of parents breeding.
  , validDNA :: [b] -> Bool -- ^ A predicate for filtering out invalid solutions during breeding.
  , breeding :: BreedingFunction b -- ^ The breeding function that should be used. See "AI.Genetic.Breeding" for possible genetic operations.
  , selection :: SelectionFunction b -- ^ The selection function determines how parents are chosen from the population for breeding. See "AI.Genetic.Selection" for options.
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

-- | Naively generates a random population, the solution type must be a member of the Random class. It is not recommended to use this method if it is likely that a large number of randomly generated will be invalid as a lot of cpu cycles will be thrown wasted generating solutions that will be thrown out.
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
  -> FitnessFunction b -- ^ A fitness function suitable for use with this library.
fitnessMap f = map (\x -> (x,f x))


-- | Runs a single step of the evolution
evolveStep :: (Eq b) => StdGen
  -> GeneticSettings b -- ^ Settings that fine tune the progression of the algorithm.
  -> FitnessFunction b
  -> [[b]] -- ^ This step's starting population
  -> (StdGen, [[b]])
evolveStep gen settings fitness initialPopulation = evolveStep' gen settings (fitness initialPopulation)

-- | Same as 'evolveStep', but the initial population already has been passed through the fitness function.
evolveStep' :: (Eq b) => StdGen
  -> GeneticSettings b -- ^ Settings that fine tune the progression of the algorithm.
  -> [([b],Fitness)] -- ^ This step's starting population
  -> (StdGen, [[b]])
evolveStep' gen settings initialPopulation = breed gen initialPopulation
  where
    populationSize = length initialPopulation

    breed gen population = (gen',population')
      where
        (gen',infEncodedPopulationList)
            = mapAccumL (\g b -> b (getParents g)) gen
            $ replicate populationSize (pickChildren . breeding settings)

        population' = take populationSize $ concat infEncodedPopulationList 

        getParents g = (g', take (maxParents settings) parents)
          where
            ((g',_), parents)
                = mapAccumL (\acc pp -> pp acc) (g,population)
                $ replicate (maxParents settings) (selection settings)

        --filters out bad children and randomly discards down to maxChildren if needed
        pickChildren (gen,children) = (gen,take (maxChildren settings) randomChildren)
          where
            validChildren = filter (validDNA settings) children
            randomChildren = if null validChildren then [] else shuffle' validChildren (length validChildren) gen

-- | Same as 'evolve' but only returns the best solution.
evolveI :: (Eq b) => StdGen
  -> GeneticSettings b -- ^ Settings for running the simulation, 'defaultSettings' are sane defaults.
  -> FitnessFunction b -- ^ The fitness function for ranking potential solutions.
  -> [[b]] -- ^ The initial population.
  -> [b] -- ^ Resulting best solution found by the simulation.
evolveI gen settings fitness initialPopulation = best
  where
    finalPopulation = evolveP gen settings fitness initialPopulation
    (best,_) = maximumBy (\(_,a) (_,b) -> compare a b) finalPopulation

-- | Evolves a population from an initial population according to the supplied settings.
evolveP :: (Eq b) => StdGen
  -> GeneticSettings b -- ^ Settings for running the simulation, 'defaultSettings' are sane defaults.
  -> FitnessFunction b -- ^ The fitness function for ranking potential solutions.
  -> [[b]] -- ^ The initial population.
  -> [([b],Fitness)] -- ^ Resulting best solution found by the simulation.
evolveP gen settings fitness initialPopulation = finalPopulation
  where
    populationSize = length initialPopulation
    finalPopulation = run 0 gen (fitness initialPopulation)

    --run :: Int -> StdGen -> [d] -> [(d,Fitness)]
    run iteration gen population
        | iteration+1 >= (maxIterations settings)       = population'' -- stop at max iterations
        | percentOverTarget >= (targetPercent settings) = population'' -- stop at found fitness
        | otherwise                                     = run (iteration+1) gen' $ population'' --keep going
          where
            (gen',population') = evolveStep' gen settings population -- next population
            population'' = fitness population'
            numOverTarget = fromIntegral $ length $ filter (\(_,f) -> f >= (targetFitness settings)) population''
            percentOverTarget = numOverTarget / (fromIntegral $ length population'')
