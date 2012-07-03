module AI.NN.Activation (
    -- | This module contains all the prewritten activation functions, and defines an encapsulating datatype.
    ActivationFunction (..)
  , activate
  , alwaysOnAF
  , linearAF
  , heavysideAF
  , sigmoidAF
  , sigmoidScaledAF
  , fastSigmoidAF
  , tanhAF
  , fastTanhAF
)
where

-- | Encapsulates derivable and underivable activation functions.
data ActivationFunction =
  -- | First value is the function, the second value is the derivative
    Derivable (Double -> Double) (Double -> Double)
  | Underivable (Double -> Double)


-- | Turns an 'ActivationFunction' in to a real valued function.
activate :: ActivationFunction -> Double -> Double
activate (Derivable d _) = d
activate (Underivable d) = d


alwaysOn :: Double -> Double
alwaysOn x = 1

alwaysOn' :: Double -> Double
alwaysOn' x = 0

-- | Used as AF for the bias neuron to allow thresholds to be trained as weights. Not terribly useful outside of that though.
alwaysOnAF :: ActivationFunction
alwaysOnAF = Derivable alwaysOn alwaysOn'


-- | Linear activation function is 'id'. Its only real use is in the input layer.
linearAF :: ActivationFunction
linearAF = Derivable id (\_ -> 1)


heavyside :: Double -> Double
heavyside x | x >= 0    = 1
            | otherwise = 0

-- | The heavyside function is 1 at x >= 0 and 0 at x < 0. The heavyside function may not be used with backpropogation training because it is not derivable. If you would like to use backpropogation consider using 'sigmoidScaledAF' with a small value for the scale.
heavysideAF = Underivable heavyside


sigmoidScaled :: Double -> Double -> Double
sigmoidScaled t x = 1 / (1 + exp (-x/t))

sigmoidScaled' :: Double -> Double -> Double
sigmoidScaled' t x = y * (1 - y)
  where y = sigmoidScaled t x

-- | A scaled version of the sigmoid function. The smaller the scale, the closer the approximation to the heavyside function.
sigmoidScaledAF ::
     Double -- ^ The amount to scale the sigmoid function.
  -> ActivationFunction
sigmoidScaledAF t = Derivable (sigmoidScaled t) (sigmoidScaled' t)


sigmoid :: Double -> Double 
sigmoid = sigmoidScaled 1

sigmoid' :: Double -> Double
sigmoid' = sigmoidScaled' 1

-- | A commonly used activation function, min value is 0, max is 1.
sigmoidAF :: ActivationFunction
sigmoidAF = Derivable sigmoid sigmoid'


fastSigmoid :: Double -> Double
fastSigmoid x' | x >= 1          = 1
               | -1 < x && x < 1 = 0.5 + (x * (1 - (abs(x) / 2)))
               | x <= -1         = 0
  where x = x' / 4.1

fastSigmoid' :: Double -> Double
fastSigmoid' x = y * (1 - y)
  where y = fastSigmoid x

-- | An approximation of the 'sigmoidAF' function that is faster to compute.
fastSigmoidAF :: ActivationFunction
fastSigmoidAF = Derivable fastSigmoid fastSigmoid'


{- Already defined in the Prelude
tanh :: Double -> Double
tanh x = 2 / (1 + exp(-2 * x)) - 1
-}

tanh' :: Double -> Double
tanh' x = 1-y*y
  where y = tanh x

-- | Another commonly used activation function, min value is -1, max is 1.
tanhAF :: ActivationFunction
tanhAF = Derivable tanh tanh'


fastTanh :: Double -> Double
fastTanh x | x > 1.92033           = 0.96016
           | 0 <= x && x <= 1.92033 = 0.96016 - 0.26037 * (x - 1.92033)^2
           | -1.92033 < x && x < 0 = 0.26037 * (x + 1.92033)^2 - 0.96016
           | x <= -1.92033         = -0.96016

fastTanh' :: Double -> Double
fastTanh' x = 1-y*y
  where y = fastTanh' x

-- | An approximation of 'tanhAF' that is faster to compute.
fastTanhAF :: ActivationFunction
fastTanhAF = Derivable fastTanh fastTanh'
