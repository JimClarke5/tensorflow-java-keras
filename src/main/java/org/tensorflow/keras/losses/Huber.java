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
public class Huber extends Loss {
    private final float delta;
    
    /**
     * Creates a Huber with Reduction.AUTO
     */
    public Huber(Ops tf) {
        this(tf, "huber_loss", Reduction.AUTO, 1.0F);
        
    }
    /**
     * Creates a Huber with Reduction.AUTO
     * @param delta the point where the Huber loss function changes from a quadratic to linear.
     */
    public Huber(Ops tf, float delta) {
        this(tf, "huber_loss", Reduction.AUTO, delta);
        
    }
    
    /**
     * Creates a Huber with Reduction.AUTO
     * 
     * @param name the name of the loss
     */
    public Huber(Ops tf, String name) {
        this(tf, name, Reduction.AUTO, 1.0F);
    }
    
    /**
     * Creates a Huber with Reduction.AUTO
     * 
     * @param name the name of the loss
     * @param delta the point where the Huber loss function changes from a quadratic to linear.
     */
    public Huber(Ops tf, String name, float delta) {
        this(tf,  name, Reduction.AUTO, delta);
    }
    
    /**
     * Creates a Huber
     * @param reduction Type of Reduction to apply to loss.
     */
    public Huber(Ops tf, Reduction reduction) {
        this(tf, "huber_loss", reduction, 1.0F);
    }
    
    /**
     * Creates a Huber
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public Huber(Ops tf, String name, Reduction reduction) {
        this(tf, name, reduction, 1.0F);
    }
    
    /**
     * Creates a Huber
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     * @param delta the point where the Huber loss function changes from a quadratic to linear.
     */
    public Huber(Ops tf, String name, Reduction reduction, float delta) {
        super(tf, name, reduction);
        this.delta = delta;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.huber(tf, labels, predictions, delta);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }
    
}
