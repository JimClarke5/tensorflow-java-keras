/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.losses;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public class MeanAbsoluteErrorTest {
    
    int index;
    float epsilon = 1e-4F;
    
    public MeanAbsoluteErrorTest() {
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
         MeanAbsoluteError instance = new MeanAbsoluteError();
         assertEquals("mean_absolute_error", instance.getName());
         
          instance = new MeanAbsoluteError("mae_1", Reduction.SUM);
          assertEquals("mae_1", instance.getName());
          assertEquals( Reduction.SUM, instance.getReduction());
          
    }

    /**
     * Test of call method, of class MeanAbsoluteError.
     */
    @Test
    public void testAllCorrectUnweighted() {
        System.out.println("testAllCorrectUnweighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                    Ops tf = Ops.create(graph).withName("test");
            MeanAbsoluteError instance = new MeanAbsoluteError();
            float[] true_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand loss = instance.call(tf, y_true, y_true);
            sess.run(loss);
            float expected = 0.0F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        index = 0;
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
        }
    }
    
    /**
     * Test of call method, of class MeanAbsoluteError.
     */
    @Test
    public void test_unweighted() {
        System.out.println("test_unweighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                    Ops tf = Ops.create(graph).withName("test");
            MeanAbsoluteError instance = new MeanAbsoluteError();
            float[] true_np = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand loss = instance.call(tf, y_true, y_pred);
            sess.run(loss);
            float expected = 5.5F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
        }
    }
    
    /**
     * Test of call method, of class MeanAbsoluteError.
     */
    @Test
    public void test_scalar_weighted() {
        System.out.println("test_scalar_weighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                    Ops tf = Ops.create(graph).withName("test");
            MeanAbsoluteError instance = new MeanAbsoluteError();
            float[] true_np = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(tf, y_true, y_pred, sampleWeight);
            sess.run(loss);
            float expected = 12.65F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
        }
    }
    
    @Test
    public void test_sample_weighted() {
        System.out.println("test_sample_weighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                    Ops tf = Ops.create(graph).withName("test");
            MeanAbsoluteError instance = new MeanAbsoluteError();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            float[] sample_narray = {1.2f, 3.4f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand sampleWeight =  tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2,1)));
            Operand loss = instance.call(tf, y_true, y_pred, sampleWeight);
            sess.run(loss);
            float expected = 81.4F / 6F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
        }
    }
    
    @Test
    public void test_zero_weighted() {
        System.out.println("test_zero_weighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                    Ops tf = Ops.create(graph).withName("test");
            MeanAbsoluteError instance = new MeanAbsoluteError();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand sampleWeight =  tf.constant(0.F);
            Operand loss = instance.call(tf, y_true, y_pred, sampleWeight);
            sess.run(loss);
            float expected = 0F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
        }
    }
    
    @Test
    public void test_timestep_weighted() {
        System.out.println("test_timestep_weighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            MeanAbsoluteError instance = new MeanAbsoluteError(Reduction.AUTO);
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            float[] sample_narray = {3f, 6f, 5f, 0f, 4f, 2f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3,1)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3,1)));
            Operand sampleWeight =  tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2,3)));
            Operand loss = instance.call(tf, y_true, y_pred, sampleWeight);
            
            sess.run(loss);
            float expected = 83F / 6F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
        }
    }
    
    @Test
    public void test_invalid_sample_weight() {
        System.out.println("test_invalid_sample_weight");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            MeanAbsoluteError instance = new MeanAbsoluteError();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            float[] sample_narray = {3f, 6f, 5f, 0f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3,1)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3,1)));
            Operand sampleWeight =  tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2,2)));
            Operand loss = instance.call(tf, y_true, y_pred, sampleWeight);
            sess.run(loss);
            float expected = 83F / 6F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
            fail();
        }catch(Exception expected) {
            
        }
    }
    
    @Test
    public void test_no_reduction() {
        System.out.println("test_no_reduction");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            MeanAbsoluteError instance = new MeanAbsoluteError(Reduction.NONE);
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand sampleWeight =  tf.constant(2.3f);
            Operand loss = instance.call(tf, y_true, y_pred, sampleWeight);
            sess.run(loss);
            final float[] expected = { 10.7333F, 14.5666F};
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                index = 0;
                result.data().scalars().forEach(f -> {
                    assertEquals(expected[index++], f.getFloat(), epsilon);
                 });
            }
        }catch(Exception expected) {
            
        }
    }
    
    @Test
    public void test_sum_reduction() {
        System.out.println("test_sum_reduction");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            MeanAbsoluteError instance = new MeanAbsoluteError(Reduction.SUM);
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand sampleWeight =  tf.constant(2.3);
            Operand loss = instance.call(tf, y_true, y_pred, sampleWeight);
            System.out.println(loss.asOutput().shape());
            sess.run(loss);
            final float[] expected = { 25.29999F };
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                index = 0;
                result.data().scalars().forEach(f -> {
                    assertEquals(expected[index++], f.getFloat(), epsilon);
                 });
            }
        }catch(Exception expected) {
            
        }
    }
    
    
}
