/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.losses;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.keras.losses.impl.LossesImpl;
import org.tensorflow.op.Ops;

/**
 *
 * @author Jim Clarke
 */
public class Losses {
    static Map<String, Supplier<Loss>> map = new HashMap<String, Supplier<Loss>>() {
        {
            //put("kld", kld::new);
            //put("kullback_leibler_divergence", kullback_leibler_divergence::new);
            //put("kldivergence", KLDivergence::new);
           // put("huber", Huber::new);
            put("mae", MeanAbsoluteError::new);
            put("mean_absolute_error", MeanAbsoluteError::new);
            put("mape", MeanAbsolutePercentageError::new);
            put("mean_absolute_percentage_error", MeanAbsolutePercentageError::new);
            put("mse", MeanSquaredError::new);
            put("mean_squared_error", MeanSquaredError::new);
            put("msle", MeanSquaredLogarithmicError::new);
            put("mean_squared_logarithmic_error", MeanSquaredLogarithmicError::new);
            put("binary_crossentropy", BinaryCrossentropy::new);
            put("categorical_crossentropy", CategoricalCrossentropy::new);
            put("categorical_hinge", CategoricalHinge::new);
            put("cosine_similarity", CosineSimilarity::new);
            put("hinge", Hinge::new);
            //put("logcosh", logcosh::new);
            //put("poisson", poisson::new);
            //put("sparse_categorical_crossentropy", sparse_categorical_crossentropy::new);
            //put("squared_hinge", squared_hinge::new);
        }
    };
    
    public static Loss get(String loss) {
        Supplier<Loss> lossFunction = map.get(loss.toLowerCase());
        return lossFunction != null ? lossFunction.get() : null;
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
        return categorical_crossentropy(tf, yTrue, yPred, false, 0.0F);
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
        return categorical_crossentropy(tf, yTrue, yPred, fromLogits, 0.0F);
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
        return categorical_crossentropy(tf, yTrue, yPred, false,labelSmoothing);
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
    public static Operand categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits, float labelSmoothing) {
        return LossesImpl.categorical_crossentropy(tf, yTrue, yPred, fromLogits, labelSmoothing);
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
     * @return the loss
     */
    public static Operand sparse_categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred) {
        return LossesImpl.sparse_categorical_crossentropy(tf, yTrue, yPred);
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
