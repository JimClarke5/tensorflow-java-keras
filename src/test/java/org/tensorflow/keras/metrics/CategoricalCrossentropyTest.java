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
public class CategoricalCrossentropyTest {
     private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public CategoricalCrossentropyTest() {
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
     * Test of call method, of class CategoricalCrossentropy.
     */
   @Test
    public void testConfig() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf,"cce", true,  2.0F, -1, TInt32.DTYPE);
            assertEquals("cce", instance.getName());
            assertEquals(2.0F, instance.getLabelSmoothing());
            assertTrue(instance.isFromLogits());
            assertEquals(TInt32.DTYPE, instance.getDataType());
        }
    }
    
    @Test
    public void testUnweighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf);
            session.run(instance.resetStates());
            int[] true_np = {0, 1, 0, 0, 0, 1};
            float[] pred_np = {0.05F, 0.95F, 0F, 0.1F, 0.8F, 0.1F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(2.3538785F, total);
            session.evaluate(2, count);
            session.evaluate(1.1769392F, result);
            
        }
    }
    
    @Test
    public void test_unweighted_with_logits() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf, true);
            session.run(instance.resetStates());
            int[] true_np = {0, 1, 0, 0, 0, 1};
            float[] pred_np = {1, 9, 0, 1, 8, 1};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(7.0022807F, total);
            session.evaluate(2, count);
            session.evaluate(3.5011404F, result);
            
        }
    }
    
    @Test
    public void test_weighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf);
            session.run(instance.resetStates());
            int[] true_np = {0, 1, 0, 0, 0, 1};
            float[] pred_np = {0.05f, 0.95f, 0f, 0.1f, 0.8f, 0.1f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            
            Operand sampleWeight = tf.constant(new float[] {1.5F, 2.F});
            Op op = instance.updateState(y_true, y_pred, sampleWeight);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            //session.print(System.out, loss);
            session.evaluate(4.6821095F, total); 
            session.evaluate(3.5, count);
            session.evaluate(1.3377455F, result);
            
        }
    }
    
    @Test
    public void test_weighted_from_logits() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf, true);
            session.run(instance.resetStates());
            int[] true_np = {0, 1, 0, 0, 0, 1};
            float[] pred_np = {1, 9, 0, 1, 8, 1};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            
            Operand sampleWeight = tf.constant(new float[] {1.5F, 2.F});
            Op op = instance.updateState(y_true, y_pred, sampleWeight);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(14.004333F, total);
            session.evaluate(3.5, count);
            session.evaluate(4.0012328F, result);
            
        }
    }
    
    @Test
    public void test_label_smoothing() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            float label_smoothing = 0.1F;
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf, true, label_smoothing);
            session.run(instance.resetStates());
            int[] true_np = {0, 1, 0, 0, 0, 1};
            float[] pred_np = {1, 9, 0, 1, 8, 1};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            
            
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(7.3356137f, total);
            session.evaluate(2, count);
            session.evaluate(3.6678069F, result);
            
        }
    }
    
}
