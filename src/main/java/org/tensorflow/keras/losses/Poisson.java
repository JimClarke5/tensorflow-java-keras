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
 * Computes the Poisson loss between the labels and predictions
 * @author Jim Clarke
 */
public class Poisson extends Loss {
    /**
     * Creates a Poisson with Reduction.AUTO
     * 
     * @param tf the TensorFlow Ops
     */
    public Poisson(Ops tf) {
        super(tf, "poisson");
    }
    
    /**
     * Creates a Poisson with Reduction.AUTO
     * 
     * @param tf the TensorFlow Ops
     * @param name the name of the loss
     */
    public Poisson(Ops tf, String name) {
        super(tf,name);
    }
    /**
     * Creates a Poisson
     * 
     * @param tf the TensorFlow Ops
     * @param reduction Type of Reduction to apply to loss.
     */
    public Poisson(Ops tf,Reduction reduction) {
        super(tf, "logcosh", reduction);
    }
    
    /**
     * Creates a Poisson
     * 
     * @param tf the TensorFlow Ops
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public Poisson(Ops tf, String name, Reduction reduction) {
        super(tf, name, reduction);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.poisson(tf, labels, predictions);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }
    
}
