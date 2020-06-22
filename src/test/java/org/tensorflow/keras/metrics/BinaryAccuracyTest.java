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
public class BinaryAccuracyTest {
    
    private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public BinaryAccuracyTest() {
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
        System.out.println("testConfig");
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            BinaryAccuracy instance = new BinaryAccuracy(tf,"bin_acc", 0.75f);
            assertEquals("bin_acc", instance.getName());
            assertEquals(0.75f, instance.getThreshold());
        }
    }
    
    @Test
    public void testCorrect() {
        System.out.println("testCorrect");
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            BinaryAccuracy instance = new BinaryAccuracy(tf);
            session.run(instance.resetStates());
            int[] true_np = {1, 0};
            float[] pred_np = {1, 0};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 1)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
            Operand result  = instance.result();
            session.evaluate(2F, total);
            session.evaluate(2, count);
            session.evaluate(1F, result);
            
        }
    }
    @Test
    public void testYPredSqueeze() {
        System.out.println("testYPredSqueeze");
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            BinaryAccuracy instance = new BinaryAccuracy(tf);
            session.run(instance.resetStates());
            int[] true_np = {1, 0};
            float[] pred_np = {1, 1};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 1, 1)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
            Operand result  = instance.result();
            session.evaluate(2F, total);
            session.evaluate(4, count);
            session.evaluate(0.5F, result);
        }
    }
    
    @Test
    public void testSampleWeight() {
        System.out.println("testSampleWeight");
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            BinaryAccuracy instance = new BinaryAccuracy(tf);
            session.run(instance.resetStates());
            int[] true_np = {2,1};
            int[] pred_np = {2,0};
            
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 1)));
            
            Operand sampleWeight = tf.reshape(tf.constant(new float[] {.5F, .2F}), tf.constant(Shape.of(2, 1)));
            Op op = instance.updateState(y_true, y_true, sampleWeight);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
            Operand result  = instance.result();
            session.evaluate(0.7F, total);
            session.evaluate(.7, count);
            session.evaluate(1.0F, result);
            
        }
    }
    
    @Test
    public void testVariableState() {
        System.out.println("testVariableState");
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            BinaryAccuracy instance = new BinaryAccuracy(tf);
            session.run(instance.resetStates());
            int[] true_np = {2,1};
            int[] pred_np = {2,0};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 1)));
            
            Operand sampleWeight = tf.reshape(tf.constant(new float[] {.5F, .2F}), tf.constant(Shape.of(2, 1)));
            Op op = instance.updateState(y_true, y_true, sampleWeight);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
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
            instance = new BinaryAccuracy(tf);
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
    
    @Test
    public void  test_binary_accuracy_threshold() {
        System.out.println("testVariableState");
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            BinaryAccuracy instance = new BinaryAccuracy(tf, 0.7f);
            session.run(instance.resetStates());
            int[] true_np = {1,1,0,0};
            float[] pred_np = {0.9f, 0.6f, 0.4f, 0.8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 1)));
            
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
            Operand result  = instance.result();
            session.evaluate(2F, total);
            session.evaluate(4, count);
            session.evaluate(0.5F, result);
            
        }
    }
    
}
