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
import org.tensorflow.types.TInt32;

/**
 *
 * @author Jim Clarke
 */
public class SquaredHingeTest {
    
     private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public SquaredHingeTest() {
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
     * Test of call method, of class SquaredHinge.
     */
    @Test
    public void testConfig() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            SquaredHinge instance = new SquaredHinge(tf,"sq_hinge", TInt32.DTYPE);
            assertEquals("sq_hinge", instance.getName());
            assertEquals(TInt32.DTYPE, instance.getDataType());
        }
    }
    
    @Test
    public void testUnweighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            SquaredHinge instance = new SquaredHinge(tf);
            session.run(instance.resetStates());
            int[] true_np = { 
                0, 1, 0, 1, 
                0, 0, 1, 1
            };
            float[] pred_np = { 
                -0.3f, 0.2f, -0.1f, 1.6f,
                -0.25f, -1.f, 0.5f, 0.6f
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
            Operand result  = instance.result();
            session.evaluate(0.72812f, total);
            session.evaluate(2, count);
            session.evaluate(0.3640625f, result);
            
            
        }
    }
    
    
    @Test
    public void test_weighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            SquaredHinge instance = new SquaredHinge(tf);
            session.run(instance.resetStates());
            int[] true_np = { 
                0, 1, 0, 1, 
                0, 0, 1, 1
            };
            float[] pred_np = { 
                -0.3f, 0.2f, -0.1f, 1.6f,
                -0.25f, -1.f, 0.5f, 0.6f
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4)));
            
            Operand sampleWeight = tf.constant(new float[] {1.5f, 2.f});
            Op op = instance.updateState(y_true, y_pred, sampleWeight);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
            Operand result  = instance.result();
            session.evaluate(1.2137499f, total);
            session.evaluate(3.5, count);
            session.evaluate(0.3467857f, result);
            
        }
    } 
}
