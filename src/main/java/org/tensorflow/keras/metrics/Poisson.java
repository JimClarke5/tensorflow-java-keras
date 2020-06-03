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
 * Computes the Poisson loss between the labels and predictions
 * @author Jim Clarke
 */
public class Poisson extends MeanMetricWrapper implements LossFunction {
    
    public static final String DEFAULT_NAME = "poisson";
    
    /**
     * Creates a Poisson 
     * 
     * @param tf the TensorFlow Ops
     */
    public Poisson(Ops tf) {
        this(tf, DEFAULT_NAME, null);
    }
    
    public Poisson(Ops tf, DataType dType) {
        this(tf, DEFAULT_NAME, dType);
    }
    
    /**
     * Creates a Poisson 
     * 
     * @param tf the TensorFlow Ops
     * @param name the name of the loss
     */
    public Poisson(Ops tf, String name) {
        this(tf,name, null);
    }
    
    public Poisson(Ops tf, String name, DataType dType) {
        super(tf,name, dType);
        super.setLoss(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        return Losses.poisson(tf, labels, predictions);
    }
    
}
