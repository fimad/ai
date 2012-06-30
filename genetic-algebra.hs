import AI.Genetic
import Data.Maybe
import System.Random

data Expr = Num Int | Plus | Minus deriving (Eq,Show)

instance Random Expr where
  random gen = case randomR (0::Int,2) gen of
    (0,gen') -> let (i,gen'') = randomR (0,100) gen' in (Num i,gen'')
    (1,gen') -> (Plus,gen')
    (2,gen') -> (Minus,gen')
  randomR _ = random

eval' :: [Expr] -> Maybe Int
eval' [] = Nothing
eval' [(Num a)] = Just a
eval' ((Num a):Plus:xs) = eval' xs >>= (\x -> Just (a+x))
eval' ((Num a):Minus:xs) = eval' xs >>= (\x -> Just (a-x))
eval' _ = Nothing
eval = eval' . reverse

validExpr :: [Expr] -> Bool
validExpr = isJust . eval

fitness goal expr = case eval expr of
  Nothing -> 0.0
  Just x -> 1.0/(exp $ log (abs (goal - fromIntegral x) + 1))

settings = defaultSettings {validDNA = validExpr}

--generates an initial population of 100 of dna length 5
initialPopulation gen = randomPopulation gen settings 5 100

main = do
  gen <- getStdGen
  let solution = evolve gen settings (fitnessMap (fitness 125)) (initialPopulation gen)
  putStrLn $ show $ solution
  putStrLn $ show $ eval solution
