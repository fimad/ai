{- | Find an expression using addition and subtraction that evaluates to N.
-}
import AI.Genetic
import AI.Genetic.Selection
import Data.Maybe
import Data.List
import System.Random

-- Code for handling expressions

data Expr = Num Int | Plus | Minus deriving (Eq,Show)

eval :: [Expr] -> Maybe Int
eval [] = Nothing
eval [(Num a)] = Just a
eval ((Num a):Plus:(Num b):xs) = eval $ (Num (a+b)):xs
eval ((Num a):Minus:(Num b):xs) = eval $ (Num (a-b)):xs
eval _ = Nothing

validExpr :: [Expr] -> Bool
validExpr = isJust . eval


-- Two ways of calculating intial populations are shown, a naive way using the generic randomPopulation and one using randomSolution which only generates valid expressions.

-- 1
instance Random Expr where
  random gen = case randomR (0::Int,2) gen of
    (0,gen') -> let (i,gen'') = randomR (0,100) gen' in (Num i,gen'')
    (1,gen') -> (Plus,gen')
    (2,gen') -> (Minus,gen')
  randomR _ = random

--2
-- faster than random _ :: Expr because it will only generate valid solutions.
randomSolution :: StdGen -> Int -> (StdGen,[Expr])
randomSolution gen 0 = (gen,[])
randomSolution gen exprSize = 
  if exprSize `mod` 2 == 1
    then (gen'', Num (i `mod` 100) : rs)
    else case i `mod` 2 of
      0 -> (gen'', Plus : rs)
      1 -> (gen'', Minus : rs)
  where
    (i,gen') = random gen
    (gen'',rs) = randomSolution gen' (exprSize-1)

infRandomPopulation :: StdGen -> Int -> [[Expr]]
infRandomPopulation gen exprSize = snd $ mapAccumL (\gen r -> r gen exprSize) gen $ repeat randomSolution

fastRandomPopulation :: StdGen -> Int -> Int -> [[Expr]]
fastRandomPopulation gen exprSize populationSize = take populationSize $ infRandomPopulation gen exprSize


-- Running the genetic search


-- | fitness function scores a potential solution from 0 to 1.
fitness goal expr = case eval expr of
  Nothing -> 0.0
  Just x -> 1.0/(exp $ log (abs (goal - fromIntegral x) + 1))

settings = defaultSettings {validDNA = validExpr, selection = tournament 15 1}

--generates an initial population of 100 of dna length 5
--initialPopulation gen = randomPopulation gen settings 9 100 -- 90% of the time is spent tossing bad solutions using this method
initialPopulation gen = fastRandomPopulation gen 5 100

main = do
  gen <- getStdGen
  let solution = evolve gen settings (fitnessMap (fitness 125)) (initialPopulation gen)
  putStrLn $ show $ solution
  putStrLn $ show $ eval solution
