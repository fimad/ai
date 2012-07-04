module AI.NN.Training (
    -- * Training Samples
    Example
--  , TrainingSamples (..)

    -- * Termination Conditions and Error Functions
  , TerminationCondition (..)
  , defaultTermination
    -- ** Total Error
  , TotalErrorFunction
  , averageError
  , accumulatedError
    -- ** Specific Error
  , SpecificErrorFunction
  , squareError
  , rootMeanSquareError
  , euclideanError

  -- * Training Methods
  , trainBackPropagation
)
where

import AI.NN
import AI.NN.Activation
import Data.Maybe
import Control.Monad
import qualified Data.Vector as V

-- | A single example of input and desired output.
type Example = (V.Vector Double, V.Vector Double)

{-
-- | A collection of 'Example's to train on.
data TrainingSamples = TrainingSamples {
    trainingSamples :: [Example] -- ^ Training samples are used directly to train the neural net
  , tuningSamples :: [Example] -- ^ Tuning samples are not directly used to train the network, but still contribute to the total error after a round of training. Holding out some data from the training set decreases the liklihood of overfitting the net to the training data.
} deriving (Show)
-}

-- | Termination conditions determine when training has been completed
data TerminationCondition = TerminationCondition {
    maxEpochs :: Maybe Int -- ^ The maximum number of iterations to run. 'Nothing' meaning that there is no bound (default is 'Just 100')
  , specificError :: SpecificErrorFunction -- ^ The function for calculating the specific error (default is 'rootMeanSquareError')
  , totalError :: TotalErrorFunction -- ^ How the specific errors are combined into a single value (default is 'accumulatedError')
  , maxTotalError :: Double -- ^ Train the network until the total error falls below this threshold (default is 0.1, but this should be changed according to the problem domain)
}

defaultTermination :: TerminationCondition
defaultTermination = TerminationCondition {
    maxEpochs = Just 100
  , specificError = rootMeanSquareError
  , totalError = accumulatedError
  , maxTotalError = 0.1
}


-- | The total error function combines the the specific errors (see 'SpecificErrorFunction') from each training sample into a single value.
type TotalErrorFunction = [Double] -> Double

-- | Averages all of the specific errors together
averageError :: TotalErrorFunction
averageError [] = 0
averageError errors = sum errors / (fromIntegral . length) errors

-- | The sum of all of the specific errors
accumulatedError :: TotalErrorFunction
accumulatedError errors = sum errors


-- | The specific error function calculates the error value for a single ideal target and actual output pair.
type SpecificErrorFunction
  =  V.Vector Double -- ^ The ideal target vector
  -> V.Vector Double -- ^ The actual output vector
  -> Double -- ^ The resulting error

-- | The square error is defined to be one half (the sum of (the squares of the differences between target and output)).
squareError :: SpecificErrorFunction
squareError target output = 0.5 * (V.foldr (\(t,o) s -> s + (t-o)^2) 0 $ V.zip target output)

-- | The root mean square error is defined to be the square root of (the sum of (the squares of the differences between the target and output) divided by the dimension of the output vector).
rootMeanSquareError :: SpecificErrorFunction
rootMeanSquareError target output = sqrt
                                  $ (V.foldr (\(t,o) s -> s + (t-o)^2) 0 $ V.zip target output)
                                  / (fromIntegral . V.length) output

-- | The euclidean error is defined to be the square root of (the sum of (the squares of the differences between the target and output)).
euclideanError :: SpecificErrorFunction
euclideanError target output = sqrt
                             $ V.foldr (\(t,o) s -> s + (t-o)^2) 0 $ V.zip target output


-- | Trains a neural net using the backpropagation algorithm. One of the requirements to using this method of training is that every neuron in the network uses an 'ActivationFunction' that has a derivitive.
trainBackPropagation ::
     TerminationCondition -- ^ Determines when the training is complete
  -> Double -- ^ The learning rate
  -> (NeuralNet -> V.Vector Double -> (V.Vector (Double,Double), V.Vector Double)) -- ^ The eval function to use
  -> [Example] -- ^ Examples to train on
  -> NeuralNet -- ^ The starting network
  -> (Int,(Double,NeuralNet))
trainBackPropagation termination learningRate eval examples nn =
      --snd . snd . head
      head
    $ dropWhile 
      (\ (i,(error,_))
        -> error > maxTotalError termination -- stop at acceptable error
        && (isNothing maxIt || fromJust maxIt > i) -- stop at max iterations
      )
    $ tail
    $ scanl (\(i,(_,nn')) epoch -> (i+1,epoch nn')) (0,(0,nn))
    $ repeat doEpochOnline
  where

    maxIt = maxEpochs termination

    doEpochOnline :: NeuralNet -> (Double,NeuralNet)
    doEpochOnline nn = (error,newNN)
      where
        errorsAndNNs = scanr (\e (error,nn) -> e nn) (0,nn) $ map trainSample examples
        newNN = snd $ head errorsAndNNs
        error = totalError termination $ map fst errorsAndNNs

    {- --These don't work correctly, and training online seems to work well enough
    doEpochOffline :: NeuralNet -> (Double,NeuralNet)
    doEpochOffline nn = (error,newNN)
      where
        runResults = map (runSample nn) examples
        error = totalError termination $ map (\(_,_,x) -> x) runResults
        outputError = foldl1 (V.zipWith (+)) $ map (\(_,x,_) -> x) runResults
        allNeurons = foldl1 (V.zipWith (\(a1,a2) (b1,b2) -> (a1+b1,a2+b2))) $ map (\(x,_,_) -> x) runResults
        newNN = adjustWeights nn allNeurons outputError

    runSample :: NeuralNet -> Example -> (V.Vector (Double,Double), V.Vector Double, Double)
    runSample nn (input,target) = (allNeurons,outputError,error)
      where
        (allNeurons,output) = eval nn input
        outputError = V.zipWith (-) target output
        error = (specificError termination) target output
    -}
    
    trainSample :: Example -> NeuralNet -> (Double,NeuralNet)
    trainSample (input,target) nn = (error,newNN)
      where
        (allNeurons,output) = eval nn input
        outputError = V.zipWith (-) target output
        error = (specificError termination) target output
        newNN = adjustWeights nn allNeurons outputError
    
    adjustWeights :: NeuralNet -> V.Vector (Double,Double) -> V.Vector Double -> NeuralNet
    adjustWeights nn allNeurons outputError =
      nn {
        weightMatrix = V.zipWith 
          (liftM2 (\w d -> w+d))
          weights 
          (V.map Just weightDeltas)
      }
      where
        weights = weightMatrix nn

        -- construct the delta for each neuron
        deltas = V.constructrN (neuronCount nn) constructDeltas :: V.Vector Double
        constructDeltas partialDelta = delta
          where
            n = (neuronCount nn - V.length partialDelta) - 1
            nextLayerIndices = layerIndices nn $ (layerForIndex nn n) + 1
            Derivable _ derivative = activationVector nn V.! n
            delta = derivative (fst $ allNeurons V.! n) * error 
              where
                neuronsBeforeOutput = sum . init . layerList $ nn
                error = if neuronsBeforeOutput <= n -- is the to neuron in the output layer?
                  then outputError V.! (n - neuronsBeforeOutput)
                  else
                    (sum . map fromJust . filter isJust)
                    $ map
                      (\l -> (weightVector nn l V.! n) >>= (\w -> Just $ w * (partialDelta V.! (l - n - 1))))
                      nextLayerIndices

        -- construct the weight delta for each weight
        weightDeltas = V.constructN (V.length weights) constructWeightDeltas
        constructWeightDeltas partialDelta = deltaWeight
          where
            from = V.length partialDelta `mod` neuronCount nn
            to = V.length partialDelta `div` neuronCount nn
            deltaWeight = learningRate * (snd $ allNeurons V.! from) * (deltas V.! to)

