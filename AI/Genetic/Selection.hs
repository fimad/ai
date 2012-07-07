module AI.Genetic.Selection (
  -- | Selection functions are responsible for choosing parents for breeding from the general population.
    SelectionFunction
  , rouletteWheel
  , tournament
)
where

import Data.List
import System.Random
import System.Random.Shuffle

-- | Takes a population that has been scored by the fitness function and chooses a member and returns the remaining population.
type SelectionFunction b = (StdGen,[([b],Double)]) -> ((StdGen,[([b],Double)]),[b]) 

-- | Roulette Wheel selection randomly choses an individual where the likelihood of being chosen is proportional to its fitness.
rouletteWheel :: SelectionFunction b
rouletteWheel (g,population) = ((g',population'), parent)
  where
    (fitnessSum, populationDistribution) = mapAccumL (\s (d,f) -> (f+s,(d,f+s))) 0 population
    (r,g') = randomR (0,fitnessSum) g
    --split the population distribution at r
    (popLessThanR, popMoreThanR) = span (\(d,f) -> f<r) populationDistribution
    --the chosen parent is the head of the list that is greater than r
    (parent,_) = head popMoreThanR
    population' = popLessThanR ++ (tail popMoreThanR)

-- | Tournament selection chooses a random subset from the population and then selects the most fit individual in the subset. See <https://en.wikipedia.org/wiki/Tournament_selection> for a more detailed description.
tournament :: (Eq b) => Int -- ^ The size of the randomly selected subset of the population.
  -> Double -- ^ Probability of choosing the most fit individual.
  -> SelectionFunction b
tournament k p (gen,population) = ((gen',population'),parent)
  where
    sortedSubSet = sortBy (\(_,a) (_,b) -> compare b a) $ take k $ shuffle' population (length population) gen
    (_,taggedSubset) = mapAccumL (\(i,s) (d,_) -> let p' = s+p*((1-p)^i) in ((i+1,p'),(d,p')) ) (0,0) sortedSubSet -- tags each member with it's probabilty instead of fitness
    (r,gen') = randomR (0.0,1.0) gen
    --split the population distribution at r
    (popLessThanR, popMoreThanR) = span (\(d,p') -> p'<r) taggedSubset
    --the chosen parent is the head of the list that is greater than r
    (parent,_) = head popMoreThanR
    population' = filter ((parent/=) . fst) population
