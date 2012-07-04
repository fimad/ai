module AI.NN (
  -- * Creating Neural Nets
  -- | Neural nets are created from 'NetDescription's which are lists of 'LayerDescription's. The helper functions 'inputLayer' and 'outputLayer' exist for creating the input and output layer descriptions. Hidden layers can be created by specifying as a tuple each component of a 'LayerDescription'. The method 'createNN' takes a 'NetDescription' and returns the corresponding 'NeuralNet'. See "AI.NN.Training" for various methods of training a network once it has been created.
    LayerDescription
  , NetDescription
  , inputLayer
  , outputLayer
  , biasNodeLayer
  , describeFeedForward
  , createNN

  -- ** Neuron Connectors
  , NeuronConnector
  , connect
  , connectNext
  , connectForward

  -- * Using Nueral Nets
  -- | After a network has been created and trained (see "AI.NN.Training") various inputs can be evaluated using 'EvalFunction's. 'VerboseEvalFunction's are identical to their less verbose counterparts but along with the resulting output also return the internal state of the network. They exist because they are needed by some training methods.
  , EvalFunction
  , VerboseEvalFunction
  , feedForward
  , feedForward'

  -- * Misc.
  -- | These functions are likely only going to be of use to those writing their own training or evaluation functions.
  , NeuralNet (..)

  , weightVector
  , layerIndices
  , layerIndices'
  , layerForIndex
  , layerForIndex'
)
where

import Data.List
import Data.Maybe
import System.Random
import AI.NN.Activation
import qualified Data.Vector as V


-- | A structure containing all the settings of a neural net including topology and weights.
data NeuralNet = NeuralNet {
    activationVector :: V.Vector ActivationFunction -- ^ n sized vector of activation functions
  , weightMatrix :: V.Vector (Maybe Double) -- ^ n*n sized vector of weights. Can be thought of as a concatination of every neuron's input weight vector.
  , layerList :: [Int] -- ^ The number of neurons in each layer
  , neuronCount :: Int -- ^ The total number of neurons
}


-- | A neuron connector is a function that sets up the output connections from every node in a layer to the neurons in the rest of the network.
type NeuronConnector = [Int] -- ^ The number of neurons in each layer
                   -> Int -- ^ The index of the from neuron
                   -> Int -- ^ The index of the to neuron
                   -> Bool -- ^ Should the two neurons be connected?


-- | A generic 'NeuronConnector' that will connect each node in the current layer to every node in the layers specified.
connect ::
     [Int] -- ^ A list of layers to connect the current layer to. Layers are given as relative indices, positive indicating forward progression through the list.
  -> NeuronConnector
connect connectedLayers layers from to = 
  if (layerForIndex' layers to - layerForIndex' layers from) `elem` connectedLayers
    then True
    else False

-- | A 'NeuronConnector' that will connect each node in the current layer to every node in the next layer.
connectNext :: NeuronConnector
connectNext = connect [1]

-- | A 'NeuronConnector' that will connect each node in the current layer to every node in every subsequent layer.
connectForward :: NeuronConnector
connectForward layers from to = connect (take (length layers - layerForIndex' layers from - 1) [1..]) layers from to


-- | Describes the layout of a single layer in a neural net. A description of a layer consists of the number of neurons in the layer, each neuron's 'ActivationFunction', and a 'NeuronConnector' which is responsible for making the output connections for each neuron in the layer.
type LayerDescription = (Int,ActivationFunction,NeuronConnector)

-- | The total description of a neural net is a list of descriptions of each layer (see 'LayerDescription').
type NetDescription = [LayerDescription]

-- | A convenience function for creating the input layer of a neural net with the specified size and connector. The activation function used is 'linearAF'.
inputLayer :: Int -> NeuronConnector -> LayerDescription
inputLayer size nc = (size,linearAF,nc)


-- | A convenience function for creating the output layer of a neural net with the specified size and 'ActivationFunction'. It uses an always 'False' function as the connector.
outputLayer :: Int -> ActivationFunction -> LayerDescription
outputLayer size af = (size,af,(\_ _ _ -> False))


-- | A bias node is a node that always outputs 1 and is connected to every node in the network. The effect of this is that the weights associated with the bias node are essentially the thresholds for every node's 'ActivationFunction' and because they are weights the can be trained as such. To use a bias node, place a 'biasNodeLayer' after the input layer in a 'NetDescription' (Remember to increase all the relative layer indices by 1 for the input layer if you do this, ie instead of connectNext you would use connect [2]).
biasNodeLayer :: LayerDescription
biasNodeLayer = (1,alwaysOnAF,connectForward)

-- | A helper function for creating common feed forward network descriptions with a bias node (see 'biasNodeLayer').
describeFeedForward ::
     Int -- ^ number of inputs
  -> Int -- ^ number of outputs
  -> [Int] -- ^ sizes of the hidden layers
  -> ActivationFunction -- ^ the activation function used by all nodes in the network
  -> NetDescription
describeFeedForward input output hiddenLayers af =
       inputLayer input (connect [2])
    :  biasNodeLayer
    :  map (\h -> (h,af,connectNext)) hiddenLayers
    ++ [outputLayer output af]

-- | Creates a neural net from a 'NetDescription'.
createNN :: (RandomGen g) => g -> NetDescription -> NeuralNet
createNN gen desc =
    NeuralNet {
        activationVector = createActivationVector
      -- we need to randomly assign weights
      , weightMatrix = V.zipWith (\w r -> w >>= (\_ -> Just r)) createWeightMatrix randomWeights
      , neuronCount = size
      , layerList = layers
    }
  where
    layers = map (\(x,_,_) -> x) desc
    size = sum layers

    createActivationVector :: V.Vector ActivationFunction
    createActivationVector = V.fromList $ concatMap (\(i,af,_) -> replicate i af) desc

    randomWeights = V.fromList $ take (size*size) $ randomRs (-1.0,1.0) gen

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
          V.singleton $ if nc layers from to then Just 0.5 else Nothing])


-- | Returns an input weight vector for a neuron at index i.
weightVector :: NeuralNet -> Int -> V.Vector (Maybe Double)
weightVector nn i = V.slice (i*size) size $ weightMatrix nn
  where size = neuronCount nn


-- | Same as 'layerIndices', but it takes a layer list instead of a neural net.
layerIndices' ::
     [Int] 
  -> Int -- ^ target layer
  -> [Int] 
layerIndices' layers l = (snd $ mapAccumL (\acc x -> (acc+x,[acc..(acc+x-1)])) 0 layers) !! l

-- | Returns a list of actual indicies for the neurons in a given layer.
layerIndices ::
     NeuralNet 
  -> Int -- ^ target layer
  -> [Int] 
layerIndices nn l = layerIndices' (layerList nn) l


-- | Same as 'layerForIndex', but it takes a layer list instead of a neural net.
layerForIndex' ::
     [Int]
  -> Int -- ^ neuron index
  -> Int -- ^ the layer that the neuron resides in
layerForIndex' layers i = length . tail $ takeWhile (i>=) $ scanl (\a b -> a+b) 0 layers

-- | Returns the layer that a specific neuron is a member of.
layerForIndex ::
     NeuralNet
  -> Int -- ^ neuron index
  -> Int -- ^ the layer that the neuron resides in
layerForIndex nn i = layerForIndex' (layerList nn) i


-- | An eval function that returns the internal state of the neural net along with the output. Verbose eval functions are needed by some training methods.
type VerboseEvalFunction =
     NeuralNet
  -> V.Vector Double -- ^ input
  -> ( V.Vector (Double,Double) , V.Vector Double) -- ^ A tuple containing (a vector containing each neuron's input and ouput as a tuple, and the output of the network)

-- | An eval function that only returns the output of the network. As a user these versions of the eval functions will likely be more useful.
type EvalFunction =
     NeuralNet
  -> V.Vector Double -- ^ input
  -> V.Vector Double -- ^ output


-- | A verbose version of 'feedForwardEval'
feedForward':: VerboseEvalFunction
feedForward' nn input = (allNeurons, outputNeurons)
  where
    size = neuronCount nn
    allNeurons = V.constructN size calcInputAndOutput 
    outputNeurons = V.drop (sum . init . layerList $ nn) $ V.map snd allNeurons

    calcInputAndOutput previous =
        if currentIndex < (head . layerList) nn -- if we are working on the input vectors, just copy the value
          then
            let
              netIn = input V.! currentIndex
            in
              (netIn, af netIn) 
          else
            let
              netIn = sum $ map (\(i,Just w) -> (snd $ previous V.! i) * w) weights
            in
              (netIn, af netIn)
      where
        currentIndex = V.length previous
        af = activate $ activationVector nn V.! currentIndex
        weights = filter (isJust . snd) $ zip [0..] $ V.toList $ weightVector nn currentIndex

-- | A fast eval function that only works on feed forward like networks with no recurrance.
feedForward :: EvalFunction
feedForward nn input = snd $ feedForward' nn input
