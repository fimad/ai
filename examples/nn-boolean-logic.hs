import AI.NN
import AI.NN.Training
import AI.NN.Activation

import Data.List
import Text.Printf
import System.Random
import qualified Data.Vector as V

-- This example shows the training of a neural net to calculate the binary functions 'xor', 'or', and 'and'

-- Examples are lists of input output pairs

xorExamples = [
      (V.fromList [1,1], V.fromList [0])
    , (V.fromList [1,0], V.fromList [0.9])
    , (V.fromList [0,1], V.fromList [0.9])
    , (V.fromList [0,0], V.fromList [0])
  ] 

orExamples = [
      (V.fromList [1,1], V.fromList [0.9])
    , (V.fromList [1,0], V.fromList [0.9])
    , (V.fromList [0,1], V.fromList [0.9])
    , (V.fromList [0,0], V.fromList [0])
  ] 

andExamples = [
      (V.fromList [1,1], V.fromList [0.9])
    , (V.fromList [1,0], V.fromList [0])
    , (V.fromList [0,1], V.fromList [0])
    , (V.fromList [0,0], V.fromList [0])
  ] 

-- partially apply the error and the eval functions
myTest = testNN rootMeanSquareError accumulatedError feedForward


-- helper functions for pretty printing examples
showVector :: V.Vector Double -> String
showVector vector =
      concat
    $ ["["]
    ++  ( intersperse ", " $ V.toList $ (V.map (printf "%.4f") vector :: V.Vector String))
    ++ ["]"]

showExample :: NeuralNet -> Example -> IO ()
showExample nn example = do
  let (input,_) = example
  let output = feedForward nn input
  printf "\t%s -> %s\n" (showVector input) (showVector output)


-- trains and displays a network
trainLogic :: String -> NeuralNet -> [Example] -> IO ()
trainLogic name nn examples = do
  putStrLn $ "Training " ++ name ++ "..."
  --trains using back propagation using the default termination conditions
  --with a learning rate of 0.8
  --using the verbose version of the feedForward evaluation method
  let trainedNN = trainBackPropagation defaultTermination 0.8 feedForward' examples nn

  --show how the net performs on the examples
  sequence_ $ map (showExample trainedNN) examples

  --Report the error
  printf "%s error = %.4f\n\n" name (myTest trainedNN examples)

main = do
  gen <- getStdGen
  -- create an initial feed forward network with 2 inputs, 1 output and 3 hidden layers with 3 nodes each
  let initialNN = createNN gen $ describeFeedForward 2 1 [3,3,3] fastTanhAF

  trainLogic "XOR" initialNN xorExamples
  trainLogic "OR" initialNN orExamples
  trainLogic "AND" initialNN andExamples

