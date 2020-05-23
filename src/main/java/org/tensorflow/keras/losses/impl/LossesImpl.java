/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.losses.impl;

import java.util.Arrays;
import java.util.function.Supplier;
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.keras.backend.K;
import org.tensorflow.keras.losses.Reduction;
import org.tensorflow.keras.utils.SmartCond;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.ReduceAll;
import org.tensorflow.op.core.ReduceMax;
import org.tensorflow.op.core.ReduceSum;
import org.tensorflow.op.core.Squeeze;
import org.tensorflow.op.nn.SoftmaxCrossEntropyWithLogits;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public class LossesImpl {
    
    // for debug
    private static Session session;
    
    private static int PRED = 0;
    private static int TRUE = 1;
    private static int WEIGHT = 1;
    
    public static Operand kullback_leibler_divergence(Ops tf, Operand yTrue, Operand yPred) {
        Operand[] ops = preamble(tf, yTrue, yPred, null);
        yPred = ops[PRED];
        yTrue = ops[TRUE];
        
        yTrue = K.clip(tf, yTrue, K.epsilon(), 1);
        yPred = K.clip(tf, yPred, K.epsilon(), 1);
        //return math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)
        return tf.reduceSum(
                tf.math.mul(yTrue, tf.math.log(tf.math.div(yTrue, yPred))),
                tf.constant(-1));
    }
    
    public static Operand mean_absolute_error(Ops tf, Operand yTrue, Operand yPred) {
        Operand[] ops = preamble(tf, yTrue, yPred, null);
        yPred = ops[PRED];
        yTrue = ops[TRUE];
        return K.mean(tf, tf.math.abs(tf.math.sub(yPred, yTrue)), tf.constant(-1));
    }
    
    public static Operand mean_absolute_percentage_error(Ops tf, Operand yTrue, Operand yPred) {
        DataType dtype = yPred.asOutput().dataType();
        Operand[] ops = preamble(tf, yTrue, yPred, null);
        yPred = ops[PRED];
        yTrue = ops[TRUE];
        Operand diff = tf.math.abs(
             tf.math.div(
                     tf.math.sub(yTrue, yPred) , 
                     K.maximum(tf,tf.math.abs(yTrue), tf.constant(K.epsilon()))
             )
        );
        return tf.math.mul(tf.dtypes.cast(tf.constant(100), dtype), K.mean(tf, diff, tf.constant(-1)));
    }
    
    public static Operand mean_squared_error(Ops tf, Operand yTrue, Operand yPred) {
        Operand[] ops = preamble(tf, yTrue, yPred, null);
        yPred = ops[PRED];
        yTrue = ops[TRUE];
        return K.mean(tf, tf.math.squaredDifference(yPred, yTrue));
        
    }
    

    
    public static Operand mean_squared_logarithmic_error(Ops tf, Operand yTrue, Operand yPred) {
        DataType dtype = yPred.asOutput().dataType();
        Operand[] ops = preamble(tf, yTrue, yPred, null);
        yPred = ops[PRED];
        yTrue = ops[TRUE];
        Operand epsilonConst = K.epsilonConstant(tf, dtype);
        Operand one = K.one(tf, dtype);
        
        Operand first_log = tf.math.log(tf.math.add(K.maximum(tf, yPred, epsilonConst), one));
        Operand second_log = tf.math.log(tf.math.add(K.maximum(tf, yTrue, epsilonConst), one));
        
        return K.mean(tf, tf.math.squaredDifference(first_log, second_log), tf.constant(-1));
        
    }
    
    
    private static Operand smoothLabelsBinaryX(Ops tf, Operand yTrue, float labelSmoothing) {
        Constant smoothing = tf.constant(labelSmoothing);
        Constant oneMinusSomoothing = tf.constant(1.F - labelSmoothing);
        Constant halfSmoothing = tf.constant(0.5F *labelSmoothing);
        return tf.math.add(  tf.math.mul(yTrue, oneMinusSomoothing), halfSmoothing);
    }
    
    public static Operand binary_crossentropy(Ops tf, Operand yTrue, Operand yPred) {
        return binary_crossentropy(tf, yTrue, yPred, false, 0.F);
    }
    public static Operand binary_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits) {
        return binary_crossentropy(tf, yTrue, yPred, fromLogits, 0.F);
    }
    
    public static Operand binary_crossentropy(Ops tf, Operand yTrue, Operand yPred, float labelSmoothing) {
        return binary_crossentropy(tf, yTrue, yPred, false,labelSmoothing);
    }
    
    public static Operand binary_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits, float labelSmoothing) {
        Operand[] ops = preamble(tf, yTrue, yPred, null);
        yPred = ops[PRED];
        yTrue = ops[TRUE];
        
        debug("yTrue", yTrue);
        debug("yPred", yPred);
        
        yTrue = SmartCond.cond(tf,
                        tf.math.notEqual(tf.constant(labelSmoothing), tf.constant(0.F)),
                        smoothLabelsBinaryX(tf, yTrue, labelSmoothing),
                        yTrue);
        debug("SmartCond/yTrue", yTrue);
        Operand bce = K.binary_crossentropy(tf, yTrue, yPred, fromLogits, session);
        debug("bce", bce);
        Operand result =  K.mean(tf, bce, tf.constant(-1));
        debug("result", result);
        return result;
    }
    
    private static Operand smoothLabelsCatX(Ops tf, Operand yTrue, float labelSmoothing) {
        Constant smoothing = tf.constant(labelSmoothing);
        Operand numClasses = tf.dtypes.cast(tf.constant(yTrue.asOutput().shape().size(1)), yTrue.asOutput().dataType());
        Constant oneMinusSomoothing = tf.constant(1.F - labelSmoothing);
        return tf.math.add(  
                tf.math.mul(yTrue, oneMinusSomoothing),
                tf.math.div(smoothing, numClasses));
    }
    
    public static Operand categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred) {
        return categorical_crossentropy(tf, yTrue, yPred, false, 0.F);
    }
    public static Operand categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits) {
        return categorical_crossentropy(tf, yTrue, yPred, fromLogits, 0.F);
    }
    
    public static Operand categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred, float labelSmoothing) {
        return categorical_crossentropy(tf, yTrue, yPred, false,labelSmoothing);
    }
    
    public static Operand categorical_crossentropy(final Ops tf,  Operand yTrue, Operand yPred, boolean fromLogits, float labelSmoothing) {
        Operand[] ops = preamble(tf, yTrue, yPred, null);
        yPred = ops[PRED];
        yTrue = ops[TRUE];
        yTrue = SmartCond.cond(tf,
                        tf.math.notEqual(tf.constant(labelSmoothing), tf.constant(0.F)),
                        smoothLabelsCatX(tf, yTrue, labelSmoothing),
                        yTrue);
        
        Op op =  K.categorical_crossentropy(tf, yTrue, yPred, fromLogits);
        if(op instanceof SoftmaxCrossEntropyWithLogits) {
            return ((SoftmaxCrossEntropyWithLogits)op).loss();
        }
        return (Operand)op;
    }

    public static Operand categorical_hinge(Ops tf, Operand yTrue, Operand yPred) {
        DataType dtype = yPred.asOutput().dataType();
        Operand[] ops = preamble(tf, yTrue, yPred, null);
        yPred = ops[PRED];
        yTrue = ops[TRUE];
        Operand pos = tf.reduceSum(tf.math.mul(yTrue,  yPred), tf.constant(-1), ReduceSum.keepDims(Boolean.FALSE));
        Operand neg = tf.reduceMax(
           tf.math.mul(
                tf.math.sub(K.one(tf, dtype), yTrue),
                yPred), 
           tf.constant(-1), ReduceMax.keepDims(Boolean.TRUE));
        
        return tf.math.maximum(
                K.zero(tf, dtype),
                tf.math.sub(neg, tf.math.add(pos, K.one(tf, dtype)))
                );
    }
    
    
    
    private static Operand l2Normalize(Ops tf, Operand  x, int axis) {
        Operand square_sum = tf.reduceSum(tf.math.square(x), tf.constant(axis),ReduceSum.keepDims(Boolean.TRUE));
        Operand x_inv_norm = tf.math.rsqrt(tf.math.maximum(square_sum, K.epsilonConstant(tf, x.asOutput().dataType())));
        return tf.math.mul(x, x_inv_norm);
        
    }
    
    public static Operand cosine_similarity(Ops tf, Operand yTrue, Operand yPred) {
        
        return cosine_similarity(tf, yTrue, yPred, -1);
    }

    public static Operand cosine_similarity(Ops tf, Operand yTrue, Operand yPred, int axis) {
        Operand[] ops = preamble(tf, yTrue, yPred, null);
        yPred = ops[PRED];
        yTrue = ops[TRUE];
        yTrue = l2Normalize(tf, yTrue, axis);
        yPred = l2Normalize(tf, yPred, axis);
        return tf.math.neg(tf.reduceSum(tf.math.mul(yTrue, yPred), tf.constant(axis),ReduceSum.keepDims(Boolean.TRUE)));
    }
    
    private static Operand maybeConvertLables(Ops tf, Operand yTrue) {
        DataType dtype = yTrue.asOutput().dataType();
        
        Operand are_zeros = tf.math.equal(yTrue, K.zero(tf, dtype));
        Operand are_ones = tf.math.equal(yTrue, K.one(tf, dtype));
        Operand is_binary = tf.reduceAll(tf.math.logicalOr(are_zeros, are_ones), tf.constant(-1),ReduceAll.keepDims(Boolean.TRUE));
        return  SmartCond.cond(tf, is_binary, 
                tf.math.sub(tf.math.mul(K.constant(tf, -2., dtype), yTrue), K.one(tf, dtype)), 
                yTrue);
        
    }

    public static Operand hinge(Ops tf, Operand yTrue, Operand yPred) {
        DataType dtype = yPred.asOutput().dataType();
        Operand[] ops = preamble(tf, yTrue, yPred, null);
        yPred = ops[PRED];
        yTrue = ops[TRUE];
        yTrue = maybeConvertLables(tf, yTrue);
        return K.mean(tf, tf.math.maximum(
                tf.math.sub(K.one(tf, dtype), 
                        tf.math.mul(yTrue, yPred)), 
                K.zero(tf, dtype)
                ),
            tf.constant(-1));
    }

    public static Operand logcosh(Ops tf, Operand yTrue, Operand yPred) {
        DataType dtype = yPred.asOutput().dataType();
        Operand[] ops = preamble(tf, yTrue, yPred, null);
        yPred = ops[PRED];
        yTrue = ops[TRUE];
        
        Operand diff = tf.math.sub(yPred, yTrue);
        Operand _logcosh = tf.math.sub(
                tf.math.add(diff,  tf.math.softplus(tf.math.mul(K.constant(tf, -2., dtype), diff))),
                tf.dtypes.cast(tf.math.log(tf.constant(2.)), dtype)
                );
        return K.mean(tf, _logcosh, tf.constant(-1));
    }

    public static Operand poisson(Ops tf, Operand yTrue, Operand yPred) {
        DataType dtype = yPred.asOutput().dataType();
        Operand[] ops = preamble(tf, yTrue, yPred, null);
        yPred = ops[PRED];
        yTrue = ops[TRUE];
        return K.mean(tf, 
            tf.math.sub(yPred, 
                    tf.math.mul(yTrue, 
                            tf.math.log(
                                    tf.math.add(yPred, K.epsilonConstant(tf, dtype))
                            )
                    )
            ),  tf.constant(-1));
    }

    // TODO
    public static Operand sparse_categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static Operand squared_hinge(Ops tf, Operand yTrue, Operand yPred) {
        DataType dtype = yPred.asOutput().dataType();
        Operand[] ops = preamble(tf, yTrue, yPred, null);
        yPred = ops[PRED];
        yTrue = ops[TRUE];
        yTrue = maybeConvertLables(tf, yTrue);
        return K.mean(tf,
            tf.math.square(tf.math.maximum(
                    tf.math.sub(K.one(tf, dtype), 
                            tf.math.mul(yTrue, yPred)),
                    K.zero(tf, dtype)))
            , tf.constant(-1));
    }
    
    public static Operand computeWeightedLoss(Ops tf, Operand losses, Reduction reduction, Operand sampleWeight) {
        DataType dtype = losses.asOutput().dataType();
        if(sampleWeight == null) {
            sampleWeight = K.one(tf, dtype);
        }
        Operand[] result = squeezeOrExpandDimensions(tf, losses, null, sampleWeight);
        losses = result[0];
        sampleWeight = result[2];
        
        Operand weighted_losses = tf.math.mul(losses, tf.dtypes.cast(sampleWeight, dtype));
        losses = reduceWeightedLoss(tf, weighted_losses, reduction);
        return tf.dtypes.cast(losses, dtype);
    }
    
    

    private static Operand reduceWeightedLoss(Ops tf, Operand weighted_losses, Reduction reduction) {
        Operand loss;
        debug("reduceWeightedLoss/weighted_losses" , weighted_losses);
        if (reduction == Reduction.NONE) {
            loss = weighted_losses;
        }else {
            loss = tf.reduceSum(weighted_losses, K.allAxis(tf, weighted_losses), ReduceSum.keepDims(Boolean.FALSE));
            //tf.reshape(loss, tf.constant(Shape.scalar()));
            debug("reduceWeightedLoss/reduceSum" , loss);
            if (reduction == Reduction.AUTO || reduction == Reduction.SUM_OVER_BATCH_SIZE) {
                loss =  safeMean(tf, loss, weighted_losses.asOutput().shape().size());
                debug("reduceWeightedLoss/safeMean" , loss);
            }
        }
        debug("reduceWeightedLoss/loss" , loss);
        return loss;
    }

    private static Operand safeMean(Ops tf, Operand losses, long numElements) {
        int rank = losses.asOutput().shape().numDimensions();
        Operand totalLoss = tf.reduceSum(losses, K.allAxis(tf, losses));
        return tf.math.divNoNan(totalLoss, tf.dtypes.cast(tf.constant(numElements), losses.asOutput().dataType()));

    }
    
    private static Operand maybeExpandWeights(Ops tf, Operand sampleWeight, Operand rankDiff) {
        return tf.select(
            tf.math.equal(rankDiff, tf.constant(-1)), 
                tf.expandDims(sampleWeight, tf.constant(-1)), sampleWeight);
    }
    

  private static Operand  maybeAdjustWeights(Ops tf,Operand sampleWeight, Operand rankDiff) {
    return tf.select(
        tf.math.equal(rankDiff, tf.constant(1)), 
            tf.squeeze(sampleWeight, Squeeze.axis(Arrays.asList(-1L))), 
            maybeExpandWeights(tf,sampleWeight, rankDiff));
  }
  
    private static Operand[] preamble(Ops tf, Operand yTrue, Operand yPred,  Operand sampleWeight){
        yTrue = tf.dtypes.cast(yTrue, yPred.asOutput().dataType());
        return squeezeOrExpandDimensions(tf, yPred, yTrue, sampleWeight);
        
    }
    private static Operand[] squeezeOrExpandDimensions(Ops tf, Operand yPred, Operand yTrue, Operand sampleWeight) {
        Operand[] result = new Operand[3]; // y_pred, y_true, sample_weight
        Shape ypredShape = yPred.asOutput().shape();
        long ypredRank = ypredShape.numDimensions();

        if (yTrue != null) {
            Shape ytrueShape = yTrue.asOutput().shape();
            long ytrueRank = ytrueShape.numDimensions();
            if (ypredRank - ytrueRank != 1 && ypredShape.size((int) ypredRank - 1) != 1) {
                //TODO
                //y_true, y_pred = confusion_matrix.remove_squeezable_dimensions(y_true, y_pred)
            }

        }
        if (sampleWeight == null) {
            result[0] = yPred;
            result[1] = yTrue;
            result[2] = null;
            return result;
        }
        Shape weightsShape = sampleWeight.asOutput().shape();
        long weightsRank = weightsShape.numDimensions();
        if (weightsRank == 0) { // scalar
            result[0] = yPred;
            result[1] = yTrue;
            result[2] = sampleWeight;
            return result;
        }
        
        if(ypredRank != Shape.UNKNOWN_SIZE && weightsRank != Shape.UNKNOWN_SIZE) {

            if (weightsRank - ypredRank == 1) {
                sampleWeight = tf.squeeze(sampleWeight);
            } else if (ypredRank - weightsRank == 1) {
                sampleWeight = tf.expandDims(sampleWeight, tf.constant(-1L));
            }
            result[0] = yPred;
            result[1] = yTrue;
            result[2] = sampleWeight;

            return result;
        }
        // Use dynamic rank.
        Operand weightsRankTensor = tf.rank(sampleWeight);
        Operand rankDiff = tf.math.sub(weightsRankTensor, tf.rank(yPred));
        sampleWeight = tf.select(
            tf.math.equal(weightsRankTensor, tf.constant(0)), 
                sampleWeight,
                maybeAdjustWeights(tf, sampleWeight, rankDiff));
        result[0] = yPred;
        result[1] = yTrue;
        result[2] = sampleWeight;

        return result;
    }
    
    private static void debug(String prefix, Operand operand) {
        if(session != null) {
            try ( Tensor<TFloat32> result = session.runner().fetch(operand).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            System.out.printf("%s:  Actual = %f\n", prefix, f.getFloat());
                         });
            }
        }
    }

    public static void setDebug(Session sess) {
        session = sess;
    }
}
