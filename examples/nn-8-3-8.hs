import AI.NN
import AI.NN.Training
import AI.NN.Activation

import Data.List
import Text.Printf
import System.Random
import qualified Data.Vector as V

-- This example shows the training of an 8 input, 8 output neural net that is intended to learn the identity function while only having a single hidden layer of 3 nodes. 

examples = map (\x -> (x,x)) [
      V.fromList [1,0,0,0,0,0,0,0]
    , V.fromList [0,1,0,0,0,0,0,0]
    , V.fromList [0,0,1,0,0,0,0,0]
    , V.fromList [0,0,0,1,0,0,0,0]
    , V.fromList [0,0,0,0,1,0,0,0]
    , V.fromList [0,0,0,0,0,1,0,0]
    , V.fromList [0,0,0,0,0,0,1,0]
    , V.fromList [0,0,0,0,0,0,0,1]

  {-
    , V.fromList [1,0,1,0,1,0,1,0]
    , V.fromList [0,1,0,1,0,1,0,1]
    , V.fromList [1,1,0,0,0,0,1,1]
    , V.fromList [0,0,1,1,1,1,0,0]
  -}
  ]

myTest = testNN rootMeanSquareError accumulatedError linearEval

-- helper functions for pretty printing examples
showVector :: V.Vector Double -> String
showVector vector =
      concat
    $ ["["]
    ++  ( intersperse ", " $ V.toList $ (V.map (show . round) vector :: V.Vector String))
    ++ ["]"]

showExample :: NeuralNet -> Example -> IO ()
showExample nn example = do
  let (input,_) = example
  let output = linearEval nn input
  printf "\t%s -> %s\n" (showVector input) (showVector output)

main = do
  gen <- getStdGen
  let initialNN = createNN gen $ describeFeedForward 8 8 [3] sigmoidAF

  let terminationCondition = defaultTermination {maxEpochs = Just 1000, maxTotalError = 0.1}
  let trainedNN = trainBackPropagation terminationCondition 0.8 linearEvalV examples initialNN

  --show how the net performs on the examples
  sequence_ $ map (showExample trainedNN) examples

  --Report the error
  printf "error = %.4f\n\n" (myTest trainedNN examples)
