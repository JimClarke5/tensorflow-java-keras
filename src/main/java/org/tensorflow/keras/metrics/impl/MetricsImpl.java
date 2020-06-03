/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.metrics.impl;

import org.tensorflow.keras.metrics.*;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import static org.tensorflow.keras.losses.impl.LossesImpl.l2Normalize;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.ReduceSum;

/**
 *
 * @author Jim Clarke
 */
public class MetricsImpl {
    
    
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
        KLDivergence instance = new KLDivergence(tf);
        Graph graph = null;
        if(tf.scope().env() instanceof Graph) {
            graph = (Graph)tf.scope().env();
        }
        initialize(graph);
        Op op =  instance.updateState(yTrue, yPred);
        run(graph, op);
        return instance.result();
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
        MeanAbsoluteError instance = new MeanAbsoluteError(tf);
        Graph graph = null;
        if(tf.scope().env() instanceof Graph) {
            graph = (Graph)tf.scope().env();
        }
        initialize(graph);
        Op op =  instance.updateState(yTrue, yPred);
        run(graph, op);
        return instance.result();
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
        MeanAbsolutePercentageError instance = new MeanAbsolutePercentageError(tf);
        Graph graph = null;
        if(tf.scope().env() instanceof Graph) {
            graph = (Graph)tf.scope().env();
        }
        initialize(graph);
        Op op =  instance.updateState(yTrue, yPred);
        run(graph, op);
        return instance.result();
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
        MeanSquaredError instance = new MeanSquaredError(tf);
        Graph graph = null;
        if(tf.scope().env() instanceof Graph) {
            graph = (Graph)tf.scope().env();
        }
        initialize(graph);
        Op op =  instance.updateState(yTrue, yPred);
        run(graph, op);
        return instance.result();
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
        MeanSquaredLogarithmicError instance = new MeanSquaredLogarithmicError(tf);
        Graph graph = null;
        if(tf.scope().env() instanceof Graph) {
            graph = (Graph)tf.scope().env();
        }
        initialize(graph);
        Op op =  instance.updateState(yTrue, yPred);
        run(graph, op);
        return instance.result();
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
        BinaryCrossentropy instance = new BinaryCrossentropy(tf, fromLogits, labelSmoothing);
        Graph graph = null;
        if(tf.scope().env() instanceof Graph) {
            graph = (Graph)tf.scope().env();
        }
        initialize(graph);
        Op op =  instance.updateState(yTrue, yPred);
        run(graph, op);
        return instance.result();
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
        CategoricalCrossentropy instance = new CategoricalCrossentropy(tf, fromLogits, labelSmoothing);
        Graph graph = null;
        if(tf.scope().env() instanceof Graph) {
            graph = (Graph)tf.scope().env();
        }
        initialize(graph);
        Op op =  instance.updateState(yTrue, yPred);
        run(graph, op);
        return instance.result();
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
        CategoricalHinge instance = new CategoricalHinge(tf);
        Graph graph = null;
        if(tf.scope().env() instanceof Graph) {
            graph = (Graph)tf.scope().env();
        }
        initialize(graph);
        Op op =  instance.updateState(yTrue, yPred);
        run(graph, op);
        return instance.result();
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
        CosineSimilarity instance = new CosineSimilarity(tf);
        Graph graph = null;
        if(tf.scope().env() instanceof Graph) {
            graph = (Graph)tf.scope().env();
        }
        initialize(graph);
        Op op =  instance.updateState(yTrue, yPred);
        run(graph, op);
        return instance.result();
    }
    
    /**
     * Computes the cosine similarity between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand cosine_proximity(Ops tf, Operand yTrue, Operand yPred) {
        return cosine_proximity(tf, yTrue, yPred, -1);
    }
    /**
     * Computes the cosine similarity between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @param axis The dimension along which the cosine similarity is computed.
     * @return the loss
     */
    public static Operand cosine_proximity(Ops tf, Operand yTrue, Operand yPred, int axis) {
        yTrue = l2Normalize(tf, yTrue, axis);
        yPred = l2Normalize(tf, yPred, axis);
        Operand mathMul = tf.math.mul(yTrue, yPred);
        Operand sum = tf.reduceSum(mathMul, tf.constant(axis),ReduceSum.keepDims(Boolean.FALSE));
        return sum;
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
        Hinge instance = new Hinge(tf);
        Graph graph = null;
        if(tf.scope().env() instanceof Graph) {
            graph = (Graph)tf.scope().env();
        }
        initialize(graph);
        Op op =  instance.updateState(yTrue, yPred);
        run(graph, op);
        return instance.result();
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
        Poisson instance = new Poisson(tf);
        Graph graph = null;
        if(tf.scope().env() instanceof Graph) {
            graph = (Graph)tf.scope().env();
        }
        initialize(graph);
        Op op =  instance.updateState(yTrue, yPred);
        run(graph, op);
        return instance.result();
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
        SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf, fromLogits, axis);
        Graph graph = null;
        if(tf.scope().env() instanceof Graph) {
            graph = (Graph)tf.scope().env();
        }
        initialize(graph);
        Op op =  instance.updateState(yTrue, yPred);
        run(graph, op);
        return instance.result();
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
        SquaredHinge instance = new SquaredHinge(tf);
        Graph graph = null;
        if(tf.scope().env() instanceof Graph) {
            graph = (Graph)tf.scope().env();
        }
        initialize(graph);
        Op op =  instance.updateState(yTrue, yPred);
        run(graph, op);
        return instance.result();
    }
    
    
    // helper functions
    
     private static void initialize(Graph graph) {
        if(graph != null) {
            try(Session session = new Session(graph)) {
                 for(Op initializer : graph.initializers()) {
                    session.runner().addTarget(initializer).run();
                }
            }
        }
    }
    
    private static void run(Graph graph, Op op) {
        if(graph != null) {
            try(Session session = new Session(graph)) {
                session.run(op);
            }
        }
    }
    

}
