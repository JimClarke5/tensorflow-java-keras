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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import static org.tensorflow.keras.metrics.impl.Reduce.COUNT;
import static org.tensorflow.keras.metrics.impl.Reduce.TOTAL;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;

/**
 *
 * @author jbclarke
 */
public class SumTest {
    private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public SumTest() {
    }
    
    @BeforeAll
    public static void setUpClass() {
    }
    
    @AfterAll
    public static void tearDownClass() {
    }
    
    @BeforeEach
    public void setUp() {
    }
    
    @AfterEach
    public void tearDown() {
    }

    /**
     * Test of call method, of class Sum.
     */
   @Test
    public void testConfig() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Sum instance = new Sum(tf,"my_sum");
            session.run(instance.initializeVars());
            assertEquals("my_sum", instance.getName());
            assertEquals(TFloat32.DTYPE, instance.getDataType());
            assertEquals(1, instance.getVariables().size());
            session.evaluate(0f, instance.getTotal());
            
            Op update = instance.updateState(tf.constant(100));
            session.run(update);
            session.evaluate(100f, instance.result());
            session.evaluate(100f, instance.getTotal());
            
            update = instance.updateState(tf.constant(new int[]{1, 5}));
            session.run(update);
            session.evaluate(106f, instance.result());
            session.evaluate(106f, instance.getTotal());
            
            session.run(instance.resetStates());
            session.evaluate(0f, instance.getTotal());
        }
    }
    
    @Test
    public void test_sum_with_sample_weight() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Sum instance = new Sum(tf, TFloat64.DTYPE);
            assertEquals(TFloat64.DTYPE, instance.getDataType());
            session.run(instance.initializeVars());
            
            //check scalar weight
            Op op = instance.updateState(tf.constant(100), tf.constant(0.5));
            session.run(op);
            Operand result  = instance.result();
            session.evaluate(50.0, instance.getTotal());
            session.evaluate(50.0, result);
            
            //  check weights not scalar and weights rank matches values rank
            op = instance.updateState(tf.constant(new float[] {1, 5}), tf.constant(new float[] {1, 0.2f}));
            session.run(op);
            result  = instance.result();
            session.evaluate(52.f, instance.getTotal());
            session.evaluate(52.f, result);
            
            // check weights broadcast
            op = instance.updateState(tf.constant(new float[] {1, 2}), tf.constant(0.5));
            session.run(op);
            result  = instance.result();
            session.evaluate(53.5f, instance.getTotal());
            session.evaluate(53.5f, result);
            
            // check weights squeeze
            op = instance.updateState(tf.constant(new float[] {1, 5}), tf.constant(new float[][] {{1}, {0.2f}}));
            session.run(op);
            result  = instance.result();
            session.evaluate(55.5f, instance.getTotal());
            session.evaluate(55.5f, result);
            
            // check weights squeeze
            op = instance.updateState(tf.constant(new float[][] {{1}, {5}}), tf.constant(new float[] {1, 0.2f}));
            session.run(op);
            result  = instance.result();
            session.evaluate(57.5f, instance.getTotal());
            session.evaluate(57.5f, result);
            
            // heck values reduced to the dimensions of weight
            op = instance.updateState(tf.constant(
                    new float[][][] {{{1.f, 2.f}, {3.f, 2.f}, {0.5f, 4.f}}}), 
                    tf.constant(new float[] {0.5f}) );
            session.run(op);
            result  = instance.result();
            session.evaluate(63.75f, instance.getTotal());
            session.evaluate(63.75f, result);
        }
    }
    
    
    
}
