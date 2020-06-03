/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.metrics;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.keras.metrics.impl.Reduce;
import org.tensorflow.op.Ops;

/**
 *
 * @author Jim Clarke
 */
public class Mean extends Reduce {

    public Mean(Ops tf) {
        this(tf, null,  null);
    }
    

    public Mean(Ops tf, DataType dType) {
        this(tf, null, dType);
    }
    

    public Mean(Ops tf,  String name, DataType dType) {
        super(tf, name, Reduction.WEIGHTED_MEAN, dType);
    }
    
}
