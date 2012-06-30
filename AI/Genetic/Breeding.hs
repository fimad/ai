module AI.Genetic.Breeding (
  -- | Each breeding function implements a simple genetic operation, and may be composed with others to create more powerful operations.
    crossover
  , mutation
)
where

import Data.List
import Control.Monad
import System.Random
import Data.Word as W
import Data.Tuple as T

-- | Performs simple k-point crossover breeding. N-1 pivots in each parent [b] are chosen, where N is the number of parents. Children are then created from all combinations of resulting parent segments.
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

