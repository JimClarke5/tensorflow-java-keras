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
import org.tensorflow.types.TInt64;

/**
 *
 * @author Jim Clarke
 */
public class CosineSimilarityTest {
     private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public CosineSimilarityTest() {
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
     * Test of call method, of class CosineSimilarity.
     */
   @Test
    public void testConfig() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf,"cosine", TInt32.DTYPE);
            assertEquals("cosine", instance.getName());
            assertEquals(TInt32.DTYPE, instance.getDataType());
        }
    }
    
    @Test
    public void testUnweighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf);
            session.run(instance.resetStates());
            float[] true_np = { 1, 9, 2, -5, -2, 6 };
            float[] pred_np = { 4, 8, 12, 8, 1, 3 };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Operand<TFloat32> loss = instance.call(y_true, y_pred, null);
            session.print(System.out, loss);
            Operand<TFloat32> total = instance.getVariable(instance.getTotalName());
            Operand<TInt64> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(0.3744381F, total);
            session.evaluate(2, count);
            session.evaluate(0.18721905F, result);
            
        }
    }
    
    
    @Test
    public void test_weighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf);
            session.run(instance.resetStates());
            float[] true_np = { 1, 9, 2, -5, -2, 6 };
            float[] pred_np = { 4, 8, 12, 8, 1, 3 };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            
            Operand sampleWeight = tf.constant(new float[] {1.2f, 3.4f});
            Op op = instance.updateState(y_true, y_pred, sampleWeight);
            session.run(op);
            Operand<TFloat32> total = instance.getVariable(instance.getTotalName());
            Operand<TInt64> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(-0.3119840621948241F, total);
            session.evaluate(4.6, count);
            session.evaluate(-0.06782262221626612F, result);
            
        }
    }
    
    @Test
    public void test_axis() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            int axis = 1;
            CosineSimilarity instance = new CosineSimilarity(tf, axis);
            session.run(instance.resetStates());
            float[] true_np = { 1, 9, 2, -5, -2, 6 };
            float[] pred_np = { 4, 8, 12, 8, 1, 3 };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Operand<TFloat32> total = instance.getVariable(instance.getTotalName());
            Operand<TInt64> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(0.3744381F, total);
            session.evaluate(2, count);
            session.evaluate(0.18721905F, result);
            
        }
    }
}
