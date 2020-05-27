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
 * Computes the hinge loss between labels and predictions.
 * @author Jim Clarke
 */
public class Hinge extends Loss {
    /**
     * Creates a Hinge with Reduction.AUTO
     */
    public Hinge(Ops tf) {
        super(tf,"hinge");
    }
    
    /**
     * Creates a Hinge with Reduction.AUTO
     * 
     * @param name the name of the loss
     */
    public Hinge(Ops tf, String name) {
        super(tf,name);
    }
    
    /**
     * Creates a Hinge
     * @param reduction Type of Reduction to apply to loss.
     */
    public Hinge(Ops tf, Reduction reduction) {
        super(tf,"hinge", reduction);
    }
    
    /**
     * Creates a Hinge
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public Hinge(Ops tf, String name, Reduction reduction) {
        super(tf, name, reduction);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.hinge(tf, labels, predictions);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }
    
}
