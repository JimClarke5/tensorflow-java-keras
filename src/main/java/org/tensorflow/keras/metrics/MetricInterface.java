/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.metrics;

import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;

/**
 *
 * @author Jim Clarke
 */
public interface MetricInterface {

    /**
     * reset states
     */
    public Op resetStates();

    /**
     * update States
     *
     * @param args Operands
     * @return the updated State
     */
    public Op updateState(Operand... args);
    
    
    /**
     * update States
     *
     * @param args Operands
     * @return the updated State
     */
    public List<Op> updateStateList(Operand... args);

    /**
     * get the result of the metric
     *
     * @return the result;
     */
    public Operand result();
    
     /**
     * get the result of the metric
     *
     * @param tf the TensorFlow Ops
     * @return the result;
     */
    public Operand result(Ops tf);
}
