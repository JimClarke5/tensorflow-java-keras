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
public class CategoricalHingeTest {
    
    int index;
    float epsilon = 1e-4F;
    
    public CategoricalHingeTest() {
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
     * Test of call method, of class CategoricalHinge.
     */
     @Test
    public void testConfig() {
         System.out.println("testConfig");
         CategoricalHinge instance = new CategoricalHinge(null);
         assertEquals("categorical_hinge", instance.getName());
         
          instance = new CategoricalHinge(null, "cat_hinge_loss", Reduction.SUM);
          assertEquals("cat_hinge_loss", instance.getName());
          assertEquals( Reduction.SUM, instance.getReduction());
          
    }
    
    /**
     * Test of call method, of class CategoricalHinge.
     */
    @Test
    public void test_reduction_none() {
        System.out.println("test_reduction_none");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                    Ops tf = Ops.create(graph).withName("test");
            CategoricalHinge instance = new CategoricalHinge(tf,Reduction.NONE);
            int[] true_np = {1, 9, 2, -5};
            float[] pred_np = {4F, 8F, 12F, 8F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,2)));
            Operand y_pred= tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,2)));
            Operand loss = instance.call(y_true, y_pred);
            sess.run(loss);
            float[] expected = { 0.0F, 65.0F};
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        index = 0;
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected[index++], f.getFloat(), epsilon);
                         });
            }
        }
    }

    /**
     * Test of call method, of class CategoricalHinge.
     */
    @Test
    public void test_unweighted() {
        System.out.println("test_unweighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                    Ops tf = Ops.create(graph).withName("test");
            CategoricalHinge instance = new CategoricalHinge(tf);
            int[] true_np = {1, 9, 2, -5};
            float[] pred_np = {4F, 8F, 12F, 8F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,2)));
            Operand y_pred= tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,2)));
            Operand loss = instance.call(y_true, y_pred);
            sess.run(loss);
            float expected = 32.5F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        index = 0;
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
        }
    }
    
    /**
     * Test of call method, of class CategoricalHinge.
     */
    @Test
    public void test_scalar_weighted() {
        System.out.println("test_scalar_weighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                    Ops tf = Ops.create(graph).withName("test");
            CategoricalHinge instance = new CategoricalHinge(tf);
            int[] true_np = {1, 9, 2, -5, -2, 6};
            float[] pred_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            sess.run(loss);
            float expected = 83.95F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
            
            Operand loss2 = instance.call(y_true, y_pred, sampleWeight);
            sess.run(loss2);
            float expected2 = 83.95F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected2, f.getFloat(), epsilon);
                         });
            }
        }
    }
    
    @Test
    public void test_sample_weighted() {
        System.out.println("test_sample_weighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                    Ops tf = Ops.create(graph).withName("test");
            CategoricalHinge instance = new CategoricalHinge(tf);
            int[] true_np = {1, 9, 2, -5, -2, 6};
            float[] pred_np = {4f, 8f, 12f, 8f, 1f, 3f};
            float[] weights_np = {1.2f, 3.4f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand sampleWeight = tf.reshape(tf.constant(weights_np), tf.constant(Shape.of(2,1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            sess.run(loss);
            float expected = 124.1F;
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
            CategoricalHinge instance = new CategoricalHinge(tf);
            int[] true_np = {1, 9, 2, -5, -2, 6};
            float[] pred_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand sampleWeight = tf.constant(0f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
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
            CategoricalHinge instance = new CategoricalHinge(tf);
            int[] true_np = {1, 9, 2, -5, -2, 6};
            float[] pred_np = {4f, 8f, 12f, 8f, 1f, 3f};
            int[] weights_np = {3, 6, 5, 0, 4, 2};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3,1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3,1)));
            Operand sampleWeight = tf.reshape(tf.constant(weights_np), tf.constant(Shape.of(2,3)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            sess.run(loss);
            float expected = 4.0F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
            
        }
    }
    
    
    
}
