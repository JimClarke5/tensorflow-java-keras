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
 * Computes the categorical hinge loss between labels and [redictions.
 * @author Jim Clarke
 */
public class CosineSimilarity extends Loss {
    private final int axis;
    /**
     * Creates a CosineSimilarity with Reduction.AUTO and axis=-1
     */
    public CosineSimilarity() {
        this("cosine_similarity", Reduction.AUTO, -1);
    }
    
    /**
     * Creates a CosineSimilarity with Reduction.AUTO
     * @param axis  The dimension along which the cosine similarity is computed.
     */
    public CosineSimilarity(int axis) {
        this("cosine_similarity", Reduction.AUTO, axis);
    }
    
    /**
     * Creates a CosineSimilarity with Reduction.AUTO  and axis=-1
     * 
     * @param name the name of the loss
     */
    public CosineSimilarity(String name) {
        this(name, Reduction.AUTO, -1);
    }
    
    /**
     * Creates a CosineSimilarity with Reduction.AUTO
     * 
     * @param name the name of the loss
     * @param axis The dimension along which the cosine similarity is computed.
     */
    public CosineSimilarity(String name, int axis) {
        this(name, Reduction.AUTO, axis);
    }
    
    /**
     * Creates a CosineSimilarity  and axis=-1
     * @param reduction Type of Reduction to apply to loss.
     */
    public CosineSimilarity(Reduction reduction) {
        this("cosine_similarity", reduction, -1);
    }
    
    /**
     * Creates a CosineSimilarity
     * @param reduction Type of Reduction to apply to loss.
     * @param axis  The dimension along which the cosine similarity is computed.
     */
    public CosineSimilarity(Reduction reduction, int axis) {
        this("cosine_similarity", reduction, axis);
    }
    
    /**
     * Creates a CosineSimilarity  and axis=-1
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public CosineSimilarity(String name, Reduction reduction) {
        this(name, reduction, -1);
    }
    
    /**
     * Creates a CosineSimilarity
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     * @param axis The dimension along which the cosine similarity is computed.
     */
    public CosineSimilarity(String name, Reduction reduction, int axis) {
        super(name, reduction);
        this.axis = axis;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Ops tf,  Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        tf = tf.withSubScope(this.getName());
        Operand losses = Losses.cosine_similarity(tf, labels, predictions);
        return super.computeWeightedLoss(tf, losses, getReduction(), sampleWeights);
    }
    
}
