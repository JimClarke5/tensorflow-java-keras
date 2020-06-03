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
 * Computes the categorical hinge loss between labels and predictions.
 * @author Jim Clarke
 */
public class CategoricalHinge extends MeanMetricWrapper implements LossFunction {
    public static final String DEFAULT_NAME = "categorical_hinge";
    
    /**
     * Creates a CategoricalHinge 
     * @param tf the TensorFlow ops
     */
    public CategoricalHinge(Ops tf) {
        this(tf, DEFAULT_NAME, null);
    }
    
    public CategoricalHinge(Ops tf, DataType dType) {
        this(tf, DEFAULT_NAME, dType);
    }
    
    /**
     * Creates a CategoricalHinge 
     * 
     * @param tf the TensorFlow ops
     * @param name the name of the loss
     */
    public CategoricalHinge(Ops tf, String name) {
        this(tf, name, null);
    }
    
    public CategoricalHinge(Ops tf, String name, DataType dType) {
        super(tf, name, dType);
        super.setLoss(this);
    }
    

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.categorical_hinge(tf, labels, predictions);
        return losses;
    }
    
}
