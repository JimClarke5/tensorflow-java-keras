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
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;

/**
 *
 * @author Jim Clarke
 */
public class CategoricalAccuracyTest {
    private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public CategoricalAccuracyTest() {
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
            CategoricalAccuracy instance = new CategoricalAccuracy(tf,"my_acc", TFloat64.DTYPE);
            assertEquals("my_acc", instance.getName());
            assertEquals(TFloat64.DTYPE, instance.getDataType());
        }
    }
    
    @Test
    public void testCorrect() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            CategoricalAccuracy instance = new CategoricalAccuracy(tf);
            session.run(instance.resetStates());
            int[] true_np = {
                0, 0, 1,
                0, 1, 0
            };
            float[] pred_np = {
                0.1f, 0.1f, 0.8f,
                0.05f, 0.95f, 0f
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(2F, total);
            session.evaluate(2, count);
            session.evaluate(1F, result);
            
        }
    }
    
    @Test
    public void testSampleWeight() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            CategoricalAccuracy instance = new CategoricalAccuracy(tf);
            session.run(instance.resetStates());
            int[] true_np = {
                0, 0, 1,
                0, 1, 0
            };
            float[] pred_np = {
                0.1f, 0.1f, 0.8f,
                0.05f, 0.95f, 0f
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            
            Operand sampleWeight = tf.reshape(tf.constant(new float[] {.5F, .2F}), tf.constant(Shape.of(2, 1)));
            Op op = instance.updateState(y_true, y_true, sampleWeight);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(0.7F, total);
            session.evaluate(.7, count);
            session.evaluate(1.0F, result);
            
        }
    }
    
    @Test
    public void testVariableState() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            CategoricalAccuracy instance = new CategoricalAccuracy(tf);
            session.run(instance.resetStates());
            int[] true_np = {
                0, 0, 1,
                0, 1, 0
            };
            float[] pred_np = {
                0.1f, 0.1f, 0.8f,
                0.05f, 0.95f, 0f
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            
            Operand sampleWeight = tf.reshape(tf.constant(new float[] {.5F, .2F}), tf.constant(Shape.of(2, 1)));
            Op op = instance.updateState(y_true, y_true, sampleWeight);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(instance.getTotalName());
            Variable<TInt32> count = instance.getVariable(instance.getCountName());
            Operand result  = instance.result();
            session.evaluate(0.7F, total);
            session.evaluate(.7, count);
            session.evaluate(1.0F, result);
            
            //2nd run
            op = instance.updateState(y_true, y_true, sampleWeight);
            session.run(op);
            result  = instance.result();
            session.evaluate(1.4F, total); 
            session.evaluate(1.4, count);
            session.evaluate(1.0F, result);
            
            // new instance same graph
            instance = new CategoricalAccuracy(tf);
            op = instance.updateState(y_true, y_true, sampleWeight);
            session.run(op);
            result  = instance.result();
            session.evaluate(2.1F, total); 
            session.evaluate(2.1, count);
            session.evaluate(1.0F, result);
            
            
            // reset variables
            session.run(instance.resetStates());
            result  = instance.result();
            op = instance.updateState(y_true, y_true, sampleWeight);
            session.run(op);
            session.evaluate(0.7F, total);
            session.evaluate(.7, count);
            session.evaluate(1.0F, result);
        }
    }
    
}
