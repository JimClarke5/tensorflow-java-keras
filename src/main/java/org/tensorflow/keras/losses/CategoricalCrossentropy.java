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
 * Computes the cross-entropy loss between true labels and predicted labels.
 *
 * @author Jim Clarke
 */
public class CategoricalCrossentropy extends Loss {

    private final boolean fromLogits;
    private final float labelSmoothing;
    private final int axis;

    /**
     * Creates a SparseCategoricalCrossentropy with Reduction.AUTO, labelSmoothing=0.0 and
     * fromLogits=false.
     */
    public CategoricalCrossentropy(Ops tf) {
        this(tf, "categorical_crossentropy", false, 0.0F, Reduction.AUTO, -1);
    }

    /**
     * Creates a SparseCategoricalCrossentropy with Reduction.AUTO, labelSmoothing=0.0 and
     * fromLogits=false.
     *
     * @param name the name of this loss function
     */
    public CategoricalCrossentropy(Ops tf, String name) {
        this(tf, name, false, 0.0F, Reduction.AUTO, -1);
    }

    /**
     * Creates a SparseCategoricalCrossentropy with labelSmoothing = 0.0 and
     * fromLogits=false.
     *
     * @param reduction Type of Reduction to apply to loss.
     */
    public CategoricalCrossentropy(Ops tf, Reduction reduction) {
        this(tf, "categorical_crossentropy", false, 0.0F, reduction, -1);
    }

    /**
     * Creates a SparseCategoricalCrossentropy with labelSmoothing = 0.0 and
     * fromLogits=false.
     *
     * @param name the name of this loss function
     * @param reduction Type of Reduction to apply to loss.
     */
    public CategoricalCrossentropy(Ops tf, String name, Reduction reduction) {
        this(tf, name, false, 0.0F, reduction, -1);
    }

     /**
     * Creates a SparseCategoricalCrossentropy with Reduction.AUTO,
     *
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     */
    public CategoricalCrossentropy(Ops tf, boolean fromLogits) {
        this(tf, "categorical_crossentropy", fromLogits, 0.0F, Reduction.AUTO, -1);
    }
    
     /**
     * Creates a SparseCategoricalCrossentropy with Reduction.AUTO,
     *
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     */
    public CategoricalCrossentropy(Ops tf, String name, boolean fromLogits) {
        this(tf, name, fromLogits, 0.0F, Reduction.AUTO, -1);
    }
    
    /**
     * Creates a SparseCategoricalCrossentropy with Reduction.AUTO,
     *
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     */
    public CategoricalCrossentropy(Ops tf, boolean fromLogits, float labelSmoothing) {
        this(tf, "categorical_crossentropy", fromLogits, labelSmoothing, Reduction.AUTO, -1);
    }

    /**
     * Creates a SparseCategoricalCrossentropy with Reduction.AUTO,
     *
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     */
    public CategoricalCrossentropy(Ops tf, String name, boolean fromLogits, float labelSmoothing) {
        this(tf, name, fromLogits, labelSmoothing, Reduction.AUTO, -1);
    }

    /**
     * Creates a SparseCategoricalCrossentropy
     *
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     * @param reduction Type of Reduction to apply to loss.
     */
    public CategoricalCrossentropy(Ops tf, boolean fromLogits, float labelSmoothing, Reduction reduction) {
        this(tf, "categorical_crossentropy", fromLogits, labelSmoothing, reduction, -1);
    }

    /**
     * Creates a SparseCategoricalCrossentropy
     *
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     * @param reduction Type of Reduction to apply to loss.
     */
    public CategoricalCrossentropy(Ops tf, String name, boolean fromLogits, float labelSmoothing, Reduction reduction, int axis) {
        super(tf, name, reduction);
        this.fromLogits = fromLogits;
        this.labelSmoothing = labelSmoothing;
        this.axis = axis;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.categorical_crossentropy(tf, labels, predictions, fromLogits, labelSmoothing, axis);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }

}
