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
public class MeanSquaredErrorTest {
     private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public MeanSquaredErrorTest() {
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

    @Test
    public void testConfig() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            MeanSquaredError instance = new MeanSquaredError(tf,"mse", TInt32.DTYPE);
            assertEquals("mse", instance.getName());
            assertEquals(TInt32.DTYPE, instance.getDataType());
        }
    }
    
    @Test
    public void testUnweighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            MeanSquaredError instance = new MeanSquaredError(tf);
            session.run(instance.resetStates());
            float[] true_np = { 
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            float[] pred_np = { 
                0, 0, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(2.0f, total);
            session.evaluate(4, count);
            session.evaluate(0.5f, result);
            
            
        }
    }
    
    
    @Test
    public void test_weighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            MeanSquaredError instance = new MeanSquaredError(tf);
            session.run(instance.resetStates());
            float[] true_np = { 
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            float[] pred_np = { 
                0, 0, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            
            Operand sampleWeight = tf.constant(new float[] {1.f, 1.5f, 2.f, 2.5f});
            Op op = instance.updateState(y_true, y_pred, sampleWeight);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(3.8f, total);
            session.evaluate(7, count);
            session.evaluate(0.542857f, result);
            
        }
    } 
}
