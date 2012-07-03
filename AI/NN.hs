module AI.NN (
    NeuralNet (..)
  , weightVector

  -- * Creating Neural Nets
  -- | Neural nets are created from 'NetDescription's which are lists of 'LayerDescription's. The helper functions 'inputLayer' and 'outputLayer' exist for creating the input and output layer descriptions. Hidden layers can be created by specifying as a tuple each component of a 'LayerDescription'. The method 'createNN' takes a 'NetDescription' and returns the corresponding 'NeuralNet'.
  , LayerDescription
  , NetDescription
  , inputLayer
  , outputLayer
  , createNN

  -- ** Neuron Connectors
  , NeuronConnector
  , feedForward

  -- * Using Nueral Nets
  , feedForwardProp
)
where

import Data.List
import Data.Maybe
import AI.NN.Activation
import qualified Data.Vector as V


--Back propogation derivation is page 90 of the nn book
--TODO Figure out a clean way of integrating the bias node. Maybe just work on getting this working, then hopefully it won't be too much trouble to extend it a bit to add a flag for that?
--a field for input layer would be useful

-- | A structure containing all the settings of a neural net including topology and weights.
data NeuralNet = NeuralNet {
    activationVector :: V.Vector ActivationFunction -- ^ n sized vector of activation functions
  , weightMatrix :: V.Vector (Maybe Double) -- ^ n*n sized vector of weights
  , layerList :: [Int] -- ^ The number of neurons in each layer
  , neuronCount :: Int -- ^ The total number of neurons
}


-- | A neuron connector is a function that sets up the output connections from every node in a layer to the neurons in the rest of the network.
type NeuronConnector = ([Int],Int,[Int]) -- ^ Which layer are we currently working on
                   -> Int -- ^ The index of the from neuron
                   -> Int -- ^ The index of the to neuron
                   -> Maybe Double -- ^ What the connection should be, 'Nothing' for no connection, while 'Just' specifies connections.

-- | Describes the layout of a single layer in a neural net. A description of a layer consists of the number of neurons in the layer, each neuron's 'ActivationFunction', and a 'NeuronConnector' which is responsible for making the output connections for each neuron in the layer.
type LayerDescription = (Int,ActivationFunction,NeuronConnector)

-- | The total description of a neural net is a list of descriptions of each layer (see 'LayerDescription').
type NetDescription = [LayerDescription]


-- | 'NeuronConnector' for feed forward neural nets. It will connect each node in the current layer to every node in the next.
feedForward :: NeuronConnector
feedForward (front,curr,b:back) from to =
  if sum front + curr <= to && to < sum front + curr + b -- if to is in the next layer
  then Just 0.5
  else Nothing
feedForward _ _ _ = Nothing -- if this is the last layer make no connections


-- | A convenience function for creating the input layer of a neural net with the specified size and connector. The activation function used is 'linearAF'.
inputLayer :: Int -> NeuronConnector -> LayerDescription
inputLayer size nc = (size,linearAF,nc)


-- | A convenience function for creating the output layer of a neural net with the specified size and 'ActivationFunction'. It uses a dummy function for the 'NeuronConnector'.
outputLayer :: Int -> ActivationFunction -> LayerDescription
outputLayer size af = (size,af,(\_ _ _ -> Nothing))


-- | Creates a neural net from a 'NetDescription'.
createNN :: NetDescription -> NeuralNet
createNN desc =
    NeuralNet {
        activationVector = createActivationVector
      , weightMatrix = createWeightMatrix
      , neuronCount = size
      , layerList = layers
    }
  where
    layers = map (\(x,_,_) -> x) desc
    size = sum layers

    createActivationVector :: V.Vector ActivationFunction
    createActivationVector = V.fromList $ concatMap (\(i,af,_) -> replicate i af) desc

    createWeightMatrix :: V.Vector (Maybe Double)
    createWeightMatrix = do
      V.concat $ (do
        to <- [0..(size-1)]
        [do
          (nc,from) <- V.fromList . concat . snd
                    $ mapAccumL
                      (\ start (i,_,nc) ->
                          ( start+i
                          , zip (repeat nc) [start..(i+start-1)]
                      ))
                      0
                      desc
          let position = getPosition from
          V.singleton $ nc position from to])

    getPosition :: Int -> ([Int],Int,[Int])
    getPosition i = (map fst front,current,map fst back)
      where
        layersAndSums = zip layers (map sum $ tail . inits $ layers)
        (front,(current,_):back) = span ((i>=) . snd) layersAndSums


{-
  weightIndex :: Int -> Int -> Int -> Int
weightIndex n from to = from*n + to
  -}

-- | Returns an input weight vector for a neuron at index i.
weightVector :: NeuralNet -> Int -> V.Vector (Maybe Double)
weightVector nn i = V.slice (i*size) size $ weightMatrix nn
  where size = neuronCount nn


-- | A fast propogation function that only works on feed forward networks.
feedForwardProp :: NeuralNet -> V.Vector Double -> V.Vector Double
feedForwardProp nn input = V.drop (sum . init . layerList $ nn) $ V.constructN size calcOutput 
  where
    size = neuronCount nn
    calcOutput previous =
        if currentIndex < (head . layerList) nn -- if we are working on the input vectors, just copy the value
          then af $ input V.! currentIndex
          else af $ sum $ map (\(i,Just w) -> (previous V.! i) * w) weights
      where
        currentIndex = V.length previous
        af = activate $ activationVector nn V.! currentIndex
        weights = filter (isJust . snd) $ zip [0..] $ V.toList $ weightVector nn currentIndex

