/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.losses;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

/**
 * The functional interface for all Loss Functions
 * @author Jim Clarke
 */
@FunctionalInterface
public interface LossFunction {
    
    /**
     *  Calculates the loss
     * 
     * @param <T> Operands extend TNumber
     * @param tf the TensorFlow Ops
     * @param labels the truth values or labels
     * @param predictions the predictions
     * @param sampleWeights Optional sample_weight acts as a coefficient 
     * for the loss. If a scalar is provided, then the loss is simply scaled 
     * by the given value. If sample_weight is a tensor of size [batch_size], 
     * then the total loss for each sample of the batch is rescaled by 
     * the corresponding element in the sample_weight vector. If the shape 
     * of sample_weight is [batch_size, d0, .. dN-1] 
     * (or can be broadcasted to this shape), then each loss element 
     * of y_pred is scaled by the corresponding value of sample_weight. 
     * (Note on dN-1: all loss functions reduce by 1 dimension, usually axis=-1.)
     * @return the loss
     */
    public  <T extends TNumber> Operand<T> call (Ops tf, Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights );
}
