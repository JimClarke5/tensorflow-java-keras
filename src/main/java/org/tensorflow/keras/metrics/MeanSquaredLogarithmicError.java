/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.metrics;

import org.tensorflow.DataType;
import org.tensorflow.keras.losses.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.metrics.impl.MeanMetricWrapper;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

/**
 * Computes the mean squared logarithmic error between labels and predictions.
 * 
 * @author Jim Clarke
 */
public class MeanSquaredLogarithmicError extends MeanMetricWrapper implements LossFunction {
    
    public static final String DEFAULT_NAME = "mean_squared_logarithmic_error";
    
    /**
     *  Creates a MeanSquaredLogarithmicError 
     */
    public MeanSquaredLogarithmicError(Ops tf) {
        this(tf,DEFAULT_NAME, null);
    }
    
    public MeanSquaredLogarithmicError(Ops tf, DataType dType) {
        this(tf, DEFAULT_NAME, dType);
    }
    
    /**
     *  Creates a MeanSquaredLogarithmicError 
     */
    public MeanSquaredLogarithmicError(Ops tf, String name) {
        this(tf, name, null);
    }
    
    public MeanSquaredLogarithmicError(Ops tf, String name, DataType dType) {
        super(tf, name, dType);
        super.setLoss(this);
    }
    
    

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.mean_squared_logarithmic_error(tf, labels, predictions);
        return losses;
    }
    
}
