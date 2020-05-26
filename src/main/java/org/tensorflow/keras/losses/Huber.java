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
    public Huber() {
        this("huber_loss", Reduction.AUTO, 1.0F);
        
    }
    /**
     * Creates a Huber with Reduction.AUTO
     * @param delta the point where the Huber loss function changes from a quadratic to linear.
     */
    public Huber(float delta) {
        this("huber_loss", Reduction.AUTO, delta);
        
    }
    
    /**
     * Creates a Huber with Reduction.AUTO
     * 
     * @param name the name of the loss
     */
    public Huber(String name) {
        this(name, Reduction.AUTO, 1.0F);
    }
    
    /**
     * Creates a Huber with Reduction.AUTO
     * 
     * @param name the name of the loss
     * @param delta the point where the Huber loss function changes from a quadratic to linear.
     */
    public Huber(String name, float delta) {
        this(name, Reduction.AUTO, delta);
    }
    
    /**
     * Creates a Huber
     * @param reduction Type of Reduction to apply to loss.
     */
    public Huber(Reduction reduction) {
        this("huber_loss", reduction, 1.0F);
    }
    
    /**
     * Creates a Huber
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public Huber(String name, Reduction reduction) {
        this(name, reduction, 1.0F);
    }
    
    /**
     * Creates a Huber
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     * @param delta the point where the Huber loss function changes from a quadratic to linear.
     */
    public Huber(String name, Reduction reduction, float delta) {
        super(name, reduction);
        this.delta = delta;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Ops tf,  Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        tf = tf.withSubScope(this.getName());
        Operand losses = Losses.huber(tf, labels, predictions, delta);
        return super.computeWeightedLoss(tf, losses, getReduction(), sampleWeights);
    }
    
}
