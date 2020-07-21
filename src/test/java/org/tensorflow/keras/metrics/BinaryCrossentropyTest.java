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
import org.tensorflow.keras.utils.TestSession.Mode;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;

/**
 *
 * @author Jim Clarke
 */
public class BinaryCrossentropyTest {
    
    private Mode tf_mode = Mode.GRAPH;
    
    public BinaryCrossentropyTest() {
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
     * Test of call method, of class BinaryCrossentropy.
     */
    @Test
    public void testConfig() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            BinaryCrossentropy instance = new BinaryCrossentropy(tf,"bce", false, 0.2F, TInt32.DTYPE);
            assertEquals("bce", instance.getName());
            assertEquals(0.2F, instance.getLabelSmoothing());
            assertEquals(TInt32.DTYPE, instance.getDataType());
        }
    }
    
    @Test
    public void testUnweighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            BinaryCrossentropy instance = new BinaryCrossentropy(tf);
            session.run(instance.resetStates());
            int[] true_np = {1, 0, 1, 0};
            float[] pred_np = {1, 1, 1, 0};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(7.666619F, total);
            session.evaluate(2, count);
            session.evaluate(3.833309F, result);
            
        }
    }
    
    @Test
    public void test_unweighted_with_logits() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            BinaryCrossentropy instance = new BinaryCrossentropy(tf, true);
            session.run(instance.resetStates());
            int[] true_np = {1, 0, 1, 0, 1, 1};
            float[] logits_np = {100.0F, -100.0F, 100.0F, 100.0F, 100.0F, -100.0F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            Op op = instance.updateState(y_true, logits);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(66.66667F, total);
            session.evaluate(2, count);
            session.evaluate(33.333332F, result);
            
        }
    }
    
    @Test
    public void test_weighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            BinaryCrossentropy instance = new BinaryCrossentropy(tf);
            session.run(instance.resetStates());
            int[] true_np = {1, 0, 1, 0};
            float[] pred_np = {1, 1, 1, 0};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            
            Operand sampleWeight = tf.constant(new float[] {1.5F, 2.F});
            Op op = instance.updateState(y_true, y_pred, sampleWeight);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            
            session.evaluate(11.499929F, total);
            session.evaluate(3.5F, count); 
            session.evaluate(3.285694F, result); 
            
        }
    }
    
    @Test
    public void test_weighted_from_logits() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            BinaryCrossentropy instance = new BinaryCrossentropy(tf, true);
            session.run(instance.resetStates());
            int[] true_np = {1, 0, 1, 0, 1, 1};
            float[] pred_np = {100.0F, -100.0F, 100.0F, 100.0F, 100.0F, -100.0F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            
            Operand sampleWeight = tf.constant(new float[] {2F, 2.5F});
            Op op = instance.updateState(y_true, y_pred, sampleWeight);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(166.66666F, total);
            session.evaluate(4.5, count);
            session.evaluate(37.037033F, result);
            
        }
    }
    
    @Test
    public void test_label_smoothing() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            float label_smoothing = 0.1F;
            BinaryCrossentropy instance = new BinaryCrossentropy(tf, true, label_smoothing);
            session.run(instance.resetStates());
            int[] true_np = {1, 0, 1};
            float[] pred_np = {100.f, -100.f, -100.f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3)));
            
            
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            float expected = 100.0F + 50.0F * label_smoothing;
            session.evaluate(35f, total);
            session.evaluate(1f, count);
            session.evaluate(35f, result);
            
        }
    }
    
}
