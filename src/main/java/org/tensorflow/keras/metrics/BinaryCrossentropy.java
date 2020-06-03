/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.metrics;

import org.tensorflow.DataType;
import org.tensorflow.keras.losses.*;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;
import org.tensorflow.keras.metrics.impl.MeanMetricWrapper;

/**
 * Computes the cross-entropy loss between true labels and predicted labels.
 *
 * @author Jim Clarke
 */
public class BinaryCrossentropy extends MeanMetricWrapper implements LossFunction {
    
    public static final String DEFAULT_NAME = "binary_crossentropy";

    private final boolean fromLogits;
    private final float labelSmoothing;

    /**
     * Creates a BinaryCrossentropy labelSmoothing=0.0 and
     * fromLogits=false.
     *
     * @param tf the TensorFlow Ops
     */
    public BinaryCrossentropy(Ops tf) {
        this(tf,DEFAULT_NAME, false, 0.0F);
    }
    
    public BinaryCrossentropy(Ops tf, DataType dType) {
        this(tf,DEFAULT_NAME, false, 0.0F,dType);
    }
    

    /**
     * Creates a BinaryCrossentropy labelSmoothing=0.0 and
     * fromLogits=false.
     *
     * @param tf the TensorFlow Ops
     * @param name the name of this loss function
     */
    public BinaryCrossentropy(Ops tf, String name) {
        this(tf, name, false, 0.0F);
    }
    
    public BinaryCrossentropy(Ops tf, String name, DataType dType) {
        this(tf, name, false, 0.0F, dType);
    }
    

  
    /**
     * Creates a BinaryCrossentropy 
     *
     * @param tf the TensorFlow Ops
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     */
    public BinaryCrossentropy(Ops tf, boolean fromLogits) {
        this(tf, DEFAULT_NAME, fromLogits, 0.0F, null);
    }
    
    public BinaryCrossentropy(Ops tf, boolean fromLogits, DataType dType) {
        this(tf, DEFAULT_NAME, fromLogits, 0.0F, dType);
    }
    

    /**
     * Creates a BinaryCrossentropy ,
     *
     * @param tf the TensorFlow Ops
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     */
    public BinaryCrossentropy(Ops tf, String name, boolean fromLogits) {
        this(tf, name, fromLogits, 0.0F, null);
    }
    public BinaryCrossentropy(Ops tf, String name, boolean fromLogits, DataType dType) {
        this(tf, name, fromLogits, 0.0F, dType);
    }

    /**
     * Creates a BinaryCrossentropy ,
     *
     * @param tf the TensorFlow Ops
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     */
    public BinaryCrossentropy(Ops tf, boolean fromLogits, float labelSmoothing) {
        this(tf, DEFAULT_NAME, fromLogits, labelSmoothing, null);
    }
    public BinaryCrossentropy(Ops tf, boolean fromLogits, float labelSmoothing, DataType dType) {
        this(tf,DEFAULT_NAME, fromLogits, labelSmoothing, dType);
    }

    /**
     * Creates a BinaryCrossentropy ,
     *
     * @param tf the TensorFlow Ops
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     */
    public BinaryCrossentropy(Ops tf, String name, boolean fromLogits, float labelSmoothing) {
        this(tf, name, fromLogits, labelSmoothing, null);
     }

    

    /**
     * Creates a BinaryCrossentropy
     *
     * @param tf the TensorFlow Ops
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     * @param dType the datatype
     */
     
    public BinaryCrossentropy(Ops tf, String name, boolean fromLogits, float labelSmoothing, DataType dType) {
        super(tf, name, dType);
        this.fromLogits = fromLogits;
        this.labelSmoothing = labelSmoothing;
        super.setLoss(this);
    }

    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.binary_crossentropy(tf, labels, predictions, fromLogits, labelSmoothing);
        return losses;
    }
    
    public boolean getFromLogits() {
        return fromLogits;
    }
    public float getLabelSmoothing () {
        return labelSmoothing;
    }
    


}
