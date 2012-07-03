module AI.NN (
    NeuralNet (..)
  , NeuronConnector
  , LayerDescription
  , NetDescription
  , createNN
  , inputLayer
  , outputLayer
  , feedForward
  , weightVector
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

--for the description function, maybe do ([Int],Int,[Int]) -> Int -> Int -> Maybe Double. where it's previous layers current layer, future layers and it needs to return a funtion that takes a from and a to and returns a weight. It will only be fed froms from the current layer, but will be fed the entire span for to's.

-- | A structure containing all the settings of a neural net including topology and weights.
data NeuralNet = NeuralNet {
    activationVector :: V.Vector ActivationFunction -- ^ n sized vector of activation functions
  , weightMatrix :: V.Vector (Maybe Double) -- ^ n*n sized vector of weights
  , layerList :: [Int] -- ^ The number of neurons in each layer
  , neuronCount :: Int -- ^ The total number of neurons
}


-- | A node connector is a function that sets up the connections for every node in a layer to the neurons in the rest of the network.
type NeuronConnector = ([Int],Int,[Int]) -- ^ Which layer are we currently working on
                   -> Int -- ^ The index of the from neuron
                   -> Int -- ^ The index of the to neuron
                   -> Maybe Double -- ^ What the connection should be, 'Nothing' for no connection, 'Just' _ to specify connections.

-- | Describes the layout of a single layer in a neural net.
type LayerDescription = (Int,ActivationFunction,NeuronConnector)

-- | The total description of a neural net is a list of each of its layers.
type NetDescription = [LayerDescription]


-- | 'NeuronConnector' for feed forward neural nets. It will connect each node in the current layer to every node in the next.
feedForward :: NeuronConnector
feedForward (front,curr,b:back) from to =
  if sum front + curr <= to && to < sum front + curr + b -- if to is in the next layer
  then Just 0.5
  else Nothing
feedForward _ _ _ = Nothing -- if this is the last layer make no connections


-- | Creates an input layer with the specified 'NeuronConnector'.
inputLayer :: Int -> NeuronConnector -> LayerDescription
inputLayer size nc = (size,linearAF,nc)


-- | Creates an output layer with the specified 'ActivationFunction'.
outputLayer :: Int -> ActivationFunction -> LayerDescription
outputLayer size af = (size,af,(\_ _ _ -> Nothing))


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

