module AI.NN.Training (
    -- * Error Functions
    ErrorFunction
  , squareError
  , rootMeanSquareError
  , euclideanError
)
where

import Data.Vector as V

-- | The type for specific error functions.
type ErrorFunction
  =  V.Vector Double -- ^ The ideal target vector
  -> V.Vector Double -- ^ The actual output vector
  -> Double -- ^ The resulting error

-- | The square error is defined to be one half (the sum of (the squares of the differences between target and output)).
squareError :: ErrorFunction
squareError target output = 0.5 * (V.foldr (\(t,o) s -> s + (t-o)^2) 0 $ V.zip target output)

-- | The root mean square error is defined to be the square root of (the sum of (the squares of the differences between the target and output) divided by the dimension of the output vector).
rootMeanSquareError :: ErrorFunction
rootMeanSquareError target output = sqrt
                                  $ (V.foldr (\(t,o) s -> s + (t-o)^2) 0 $ V.zip target output)
                                  / (fromIntegral . V.length) output

-- | The euclidean error is defined to be the square root of (the sum of (the squares of the differences between the target and output)).
euclideanError :: ErrorFunction
euclideanError target output = sqrt
                             $ V.foldr (\(t,o) s -> s + (t-o)^2) 0 $ V.zip target output
