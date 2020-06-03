/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.losses;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.keras.losses.impl.LossesImpl;
import org.tensorflow.op.Ops;

/**
 *
 * @author Jim Clarke
 */
public class Losses {
    static Map<String, Function<Ops, Loss >> map = new HashMap<String, Function<Ops, Loss >>() {
        {
            put("kld", tf -> new KLDivergence(tf));
            put("kullback_leibler_divergence", tf -> new KLDivergence(tf));
            put("kldivergence", tf -> new KLDivergence(tf));
            put("huber", tf -> new Huber(tf));
            put("mae", tf -> new MeanAbsoluteError(tf));
            put("mean_absolute_error", tf -> new MeanAbsoluteError(tf));
            put("mape", tf -> new MeanAbsolutePercentageError(tf));
            put("mean_absolute_percentage_error", tf -> new MeanAbsolutePercentageError(tf));
            put("mse", tf -> new MeanSquaredError(tf));
            put("mean_squared_error", tf -> new MeanSquaredError(tf));
            put("msle", tf -> new MeanSquaredLogarithmicError(tf));
            put("mean_squared_logarithmic_error", tf -> new MeanSquaredLogarithmicError(tf));
            put("binary_crossentropy", tf -> new BinaryCrossentropy(tf));
            put("categorical_crossentropy", tf -> new CategoricalCrossentropy(tf));
            put("categorical_hinge", tf -> new CategoricalHinge(tf));
            put("cosine_similarity", tf -> new CosineSimilarity(tf));
            put("hinge", tf -> new Hinge(tf));
            put("logcosh", tf -> new LogCosh(tf));
            put("poisson", tf -> new Poisson(tf));
            put("sparse_categorical_crossentropy", tf -> new SparseCategoricalCrossentropy(tf));
            put("squared_hinge", tf -> new SquaredHinge(tf));
        }
    };
    
    /**
     * Get a Loss
     *
     * @param tf The TensorFlow Ops
     * @param lossFunction either a String that identifies the
     * Loss, an Loss class, or an Loss object.
     * @return the loss object or null if not found.
     */
    public static Loss get(Ops tf, Object lossFunction) {
        return get(tf, lossFunction, null);
    }

    /**
     * Get a Loss based on a lambda of the form: 
     * (Ops ops) -> create(Ops ops) 
     *
     * @param tf The TensorFlow Ops
     * @param lambda a lambda function
     * @return the Intializer object
     */
    public static Loss get(Ops tf, Function<Ops, Loss > lambda) {
        return lambda.apply(tf);
    }
    
      /**
      * Get a Loss  based on a lambda of the form:  () -> create() 
      * @param lambda a lambda function
      * @return the Intializer object
      */
    public static Loss get( Supplier<Loss > lambda) {
         return lambda.get();
    }

    /**
     * Get a Loss
     *
     * @param tf The TensorFlow Ops
     * @param lossFunction
     * @param custom_functions a map of Loss lambdas that will be queried
     * if the loss is not found in the standard keys
     * @return the Loss object
     */
    public static Loss get(Ops tf, Object lossFunction, Map<String, Function<Ops, Loss > > custom_functions) {
        if (lossFunction != null) {
            if (lossFunction instanceof String) {
                String s = lossFunction.toString(); // do this for Java 8 rather than Pattern Matching for instanceof
                Function<Ops, Loss > function = map.get(s);
                if (function == null && custom_functions != null) {
                    function = custom_functions.get(s);
                }
                return function != null ? function.apply(tf) : null;
            } else if (lossFunction instanceof Class) {
                Class c = (Class) lossFunction; // do this for Java 8 rather than Pattern Matching for instanceof
                try {
                    Constructor ctor = c.getConstructor(Ops.class);
                    return (Loss)ctor.newInstance(tf);
                } catch (NoSuchMethodException | InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException ex) {
                    Logger.getLogger(Losses.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else if (lossFunction instanceof Loss) {
                return (Loss) lossFunction; // do this for Java 8 rather than Pattern Matching for instanceof
            }
        } else {
            return null;
        }

        throw new IllegalArgumentException(
                "lossFunction must be a symbolic name, Loss, lamda or a Class object");
    }

    

    /**
     * Computes Kullback-Leibler divergence loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     *
     * @return the loss
     */
    public static Operand KLD(Ops tf, Operand yTrue, Operand yPred) {
        return kullback_leibler_divergence(tf, yTrue, yPred);
    }

    /**
     * Computes Kullback-Leibler divergence loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand kld(Ops tf, Operand yTrue, Operand yPred) {
        return kullback_leibler_divergence(tf, yTrue, yPred);
    }

    /**
     * Computes Kullback-Leibler divergence loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand kullback_leibler_divergence(Ops tf, Operand yTrue, Operand yPred) {
        return LossesImpl.kullback_leibler_divergence(tf, yTrue, yPred);

    }

    /**
     * Computes the mean absolute error between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand MAE(Ops tf, Operand yTrue, Operand yPred) {
        return mean_absolute_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean absolute error between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand mae(Ops tf, Operand yTrue, Operand yPred) {
        return mean_absolute_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean absolute error between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand mean_absolute_error(Ops tf, Operand yTrue, Operand yPred) {
        return LossesImpl.mean_absolute_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean absolute percentage error between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue
     * @param yPred
     * @return the loss
     */
    public static Operand MAPE(Ops tf, Operand yTrue, Operand yPred) {
        return mean_absolute_percentage_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean absolute percentage error between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand mape(Ops tf, Operand yTrue, Operand yPred) {
        return mean_absolute_percentage_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean absolute percentage error between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand mean_absolute_percentage_error(Ops tf, Operand yTrue, Operand yPred) {
        return LossesImpl.mean_absolute_percentage_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean squared error between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand MSE(Ops tf, Operand yTrue, Operand yPred) {
        return mean_squared_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean squared error between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand mse(Ops tf, Operand yTrue, Operand yPred) {
        return mean_squared_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean squared error between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand mean_squared_error(Ops tf, Operand yTrue, Operand yPred) {
        return LossesImpl.mean_squared_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean squared logarithmic error between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand MSLE(Ops tf, Operand yTrue, Operand yPred) {
        return mean_squared_logarithmic_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean squared logarithmic error between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand msle(Ops tf, Operand yTrue, Operand yPred) {
        return mean_squared_logarithmic_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean squared logarithmic error between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand mean_squared_logarithmic_error(Ops tf, Operand yTrue, Operand yPred) {
        return LossesImpl.mean_squared_logarithmic_error(tf, yTrue, yPred);
    }
    
    /**
     * Computes the binary crossentropy loss.
     * 
     * @param tf the TensorFlow Ops
     * @param yTrue  true targets
     * @param yPred the predictions
     * @return the loss
     */
    public static Operand binary_crossentropy(Ops tf, Operand yTrue, Operand yPred) {
        return binary_crossentropy(tf, yTrue, yPred, false, 0.0F);
        
    }
    
    /**
     * Computes the binary crossentropy loss.
     * 
     * @param tf the TensorFlow Ops
     * @param yTrue  true targets
     * @param yPred the predictions
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @return the loss
     */
    public static Operand binary_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits) {
        return binary_crossentropy(tf, yTrue, yPred, fromLogits, 0.0F);
    }
    
    /**
     *  Computes the binary crossentropy loss.
     * 
     * @param tf the TensorFlow Ops
     * @param yTrue  true targets
     * @param yPred the predictions
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. 
     * When > 0, we compute the loss between the predicted labels and 
     * a smoothed version of the true labels, where the smoothing squeezes
     * the labels towards 0.5. Larger values of label_smoothing correspond 
     * to heavier smoothing.
     * @return the loss
     */
    public static Operand binary_crossentropy(Ops tf, Operand yTrue, Operand yPred,  float labelSmoothing) {
        return binary_crossentropy(tf, yTrue, yPred, false, labelSmoothing);
    }

    /**
     * Computes the binary crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred the predictions
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. 
     * When > 0, we compute the loss between the predicted labels and 
     * a smoothed version of the true labels, where the smoothing squeezes
     * the labels towards 0.5. Larger values of label_smoothing correspond 
     * to heavier smoothing.
     * @return the loss
     */
    public static Operand binary_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits, float labelSmoothing) {
        return LossesImpl.binary_crossentropy(tf, yTrue, yPred, fromLogits, labelSmoothing);
    }
    
     /**
     * Computes the categorical crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred) {
        return categorical_crossentropy(tf, yTrue, yPred, false, 0.0F, -1);
    }
    
     /**
     * Computes the categorical crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     *  @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @return the loss
     */
    public static Operand categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits) {
        return categorical_crossentropy(tf, yTrue, yPred, fromLogits, 0.0F, -1);
    }
    
     /**
     * Computes the categorical crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. 
     * When > 0, we compute the loss between the predicted labels and 
     * a smoothed version of the true labels, where the smoothing squeezes
     * the labels towards 0.5. Larger values of label_smoothing correspond 
     * to heavier smoothing.
     * @return the loss
     */
    public static Operand categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred, float labelSmoothing) {
        return categorical_crossentropy(tf, yTrue, yPred, false,labelSmoothing, -1);
    }

    /**
     * Computes the categorical crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     *  @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. 
     * When > 0, we compute the loss between the predicted labels and 
     * a smoothed version of the true labels, where the smoothing squeezes
     * the labels towards 0.5. Larger values of label_smoothing correspond 
     * to heavier smoothing.
     * @return the loss
     */
    public static Operand categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits, float labelSmoothing, int axis) {
        return LossesImpl.categorical_crossentropy(tf, yTrue, yPred, fromLogits, labelSmoothing,axis);
    }

    /**
     * Computes the categorical hinge loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand categorical_hinge(Ops tf, Operand yTrue, Operand yPred) {
        return LossesImpl.categorical_hinge(tf, yTrue, yPred);
    }

    /**
     * Computes the cosine similarity between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand cosine_similarity(Ops tf, Operand yTrue, Operand yPred) {
        return LossesImpl.cosine_similarity(tf, yTrue, yPred);
    }

    /**
     * Computes the hinge loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand hinge(Ops tf, Operand yTrue, Operand yPred) {
        return LossesImpl.hinge(tf, yTrue, yPred);
    }
    /**
     * Computes the hinge loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @param delta
     * @return the loss
     */
    public static Operand huber(Ops tf, Operand yTrue, Operand yPred, float delta) {
        return LossesImpl.huber(tf, yTrue, yPred, delta);
    }

    /**
     * Logarithm of the hyperbolic cosine of the prediction error.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand logcosh(Ops tf, Operand yTrue, Operand yPred) {
        return LossesImpl.logcosh(tf, yTrue, yPred);
    }

    /**
     * Computes the Poisson loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand poisson(Ops tf, Operand yTrue, Operand yPred) {
        return LossesImpl.poisson(tf, yTrue, yPred);
    }
    
     /**
     * Computes the sparse categorical crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param axis The dimension along which the entropy is computed.
     * @return the loss
     */
    public static Operand sparse_categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits) {
        return sparse_categorical_crossentropy(tf, yTrue, yPred, fromLogits, -1);
    }

    /**
     * Computes the sparse categorical crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param axis The dimension along which the entropy is computed.
     * @return the loss
     */
    public static Operand sparse_categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits, int axis) {
        return LossesImpl.sparse_categorical_crossentropy(tf, yTrue, yPred, fromLogits, axis);
    }

    /**
     * Computes the squared hinge loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand squared_hinge(Ops tf, Operand yTrue, Operand yPred) {
        return LossesImpl.squared_hinge(tf, yTrue, yPred);
    }
    
    public static void setDebug(Session session) {
        LossesImpl.setDebug(session);
    }

}
