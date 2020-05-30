/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.utils;

import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import org.tensorflow.DataType;
import org.tensorflow.EagerSession;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TNumber;

/**
 *
 * @author Jim Clarke
 */
public class GraphTestSession extends TestSession {
    private final Graph graph;
    private final Session session;
    private final Ops tf;
    
    public GraphTestSession() {
        graph = new Graph();  
        session = new Session(graph);
        tf = Ops.create(graph).withName("test");
    }

    @Override
    public Ops getTF() {
        return tf;
    }
    
    public Graph getGraph() {
        return graph;
    }
    
    public Session getSession() {
        return session;
    }

    @Override
    public void close()  {
        session.close();
        graph.close();
    }

    @Override
    public boolean isEager() {
        return false;
    }
    
    @Override
    public Session getGraphSession() {
        return this.session;
    }

    @Override
    public EagerSession getEagerSession() {
        return null;
    }
    
    @Override
    public <T extends TNumber> void evaluate(Number[] expected, Operand<T> input) {
        boolean scalarExpected = expected.length == 1;
        this.getGraphSession().run(input);
        
        
        DataType dtype = input.asOutput().dataType();
        if(dtype == TFloat32.DTYPE) {
            AtomicInteger index = new AtomicInteger();
            if(debug) {
                try ( Tensor<TFloat32> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TFloat32.DTYPE)) {
                    result.data().scalars().forEach(f -> {
                        System.out.printf("%d). %f\n", index.incrementAndGet(), f.getFloat());
                    });
                }
            }
            index.set(0);
            try ( Tensor<TFloat32> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TFloat32.DTYPE)) {
                    result.data().scalars().forEach(f -> {
                        assertEquals(expected[index.get()].floatValue(), f.getFloat(), epsilon);
                        if(!scalarExpected)
                       index.incrementAndGet();
                    });
            }
        }else if(dtype == TFloat64.DTYPE) {
            AtomicInteger index = new AtomicInteger();
                       if(debug) {
                try ( Tensor<TFloat64> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TFloat64.DTYPE)) {
                    result.data().scalars().forEach(f -> {
                        System.out.printf("%d). %f\n", index.incrementAndGet(), f.getDouble());
                    });
                }
            }
            index.set(0);
            try ( Tensor<TFloat64> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TFloat64.DTYPE)) {
                    result.data().scalars().forEach(f -> {
                        assertEquals(expected[index.get()].doubleValue(), f.getDouble(), epsilon);
                        if(!scalarExpected)
                       index.incrementAndGet();
                    });
            }
        }else if(dtype == TInt32.DTYPE) {
            AtomicInteger index = new AtomicInteger();
                       if(debug) {
                try ( Tensor<TInt32> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TInt32.DTYPE)) {
                    result.data().scalars().forEach(f -> {
                        System.out.printf("%d). %d\n", index.incrementAndGet(), f.getInt());
                    });
                }
            }
            index.set(0);
            try ( Tensor<TInt32> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TInt32.DTYPE)) {
                    result.data().scalars().forEach(f -> {
                        assertEquals(expected[index.get()].intValue(), f.getInt());
                        if(!scalarExpected)
                       index.incrementAndGet();
                    });
            }
        } else if(dtype == TInt64.DTYPE) {
            Operand<TInt64> o = (Operand<TInt64>)input;
            AtomicInteger index = new AtomicInteger();
            if(debug) {
                try ( Tensor<TInt64> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TInt64.DTYPE)) {
                    result.data().scalars().forEach(f -> {
                        System.out.printf("%d). %d\n", index.incrementAndGet(), f.getLong());
                    });
                }
            }
            index.set(0);
            try ( Tensor<TInt64> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TInt64.DTYPE)) {
                    result.data().scalars().forEach(f -> {
                        assertEquals(expected[index.get()].longValue(), f.getLong());
                        if(!scalarExpected)
                       index.incrementAndGet();
                    });
            }
        }else {
            fail("Unexpected DataType: " + dtype);
        }
    }
    
    @Override
    public <T extends TNumber> void evaluate(Operand<T> expected, Operand<T> input) {
        this.getGraphSession().run(expected);
        this.getGraphSession().run(input);
        
        assert(input.asTensor().shape().equals(expected.asTensor().shape()));
        
        DataType dtype = input.asOutput().dataType();
        if(dtype == TFloat32.DTYPE) {
            AtomicInteger index = new AtomicInteger();
            final Operand<TFloat32> finalExpected = (Operand<TFloat32>)expected;
            if(debug) {
                try ( Tensor<TFloat32> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TFloat32.DTYPE)) {
                    result.data().scalars().forEachIndexed((idx,f) -> {
                        System.out.printf("%d). %f <==> %f\n", index.incrementAndGet(), finalExpected.data().getFloat(idx), f.getFloat());
                    });
                }
            }
            index.set(0);
            try ( Tensor<TFloat32> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TFloat32.DTYPE)) {
                    result.data().scalars().forEachIndexed((idx, f) -> {
                        assertEquals(finalExpected.data().getFloat(idx), f.getFloat(), epsilon);
                    });
            }
        }else if(dtype == TFloat64.DTYPE) {
                        AtomicInteger index = new AtomicInteger();
            final Operand<TFloat64> finalExpected = (Operand<TFloat64>)expected;
            if(debug) {
                try ( Tensor<TFloat64> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TFloat64.DTYPE)) {
                    result.data().scalars().forEachIndexed((idx,f) -> {
                        System.out.printf("%d). %f <==> %f\n", index.incrementAndGet(), finalExpected.data().getDouble(idx), f.getDouble());
                    });
                }
            }
            index.set(0);
            try ( Tensor<TFloat64> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TFloat64.DTYPE)) {
                    result.data().scalars().forEachIndexed((idx, f) -> {
                        assertEquals(finalExpected.data().getDouble(idx), f.getDouble(), epsilon);
                    });
            }
        }else if(dtype == TInt32.DTYPE) {
                        AtomicInteger index = new AtomicInteger();
            final Operand<TInt32> finalExpected = (Operand<TInt32>)expected;
            if(debug) {
                try ( Tensor<TInt32> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TInt32.DTYPE)) {
                    result.data().scalars().forEachIndexed((idx,f) -> {
                        System.out.printf("%d). %f <==> %f\n", index.incrementAndGet(), finalExpected.data().getInt(idx), f.getInt());
                    });
                }
            }
            index.set(0);
            try ( Tensor<TInt32> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TInt32.DTYPE)) {
                    result.data().scalars().forEachIndexed((idx, f) -> {
                        assertEquals(finalExpected.data().getInt(idx), f.getInt());
                    });
            }
        } else if(dtype == TInt64.DTYPE) {
                        AtomicInteger index = new AtomicInteger();
            final Operand<TInt64> finalExpected = (Operand<TInt64>)expected;
            if(debug) {
                try ( Tensor<TInt64> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TInt64.DTYPE)) {
                    result.data().scalars().forEachIndexed((idx,f) -> {
                        System.out.printf("%d). %f <==> %f\n", index.incrementAndGet(), finalExpected.data().getLong(idx), f.getLong());
                    });
                }
            }
            index.set(0);
            try ( Tensor<TInt64> result = this.getGraphSession().runner().fetch(input).run().get(0).expect(TInt64.DTYPE)) {
                    result.data().scalars().forEachIndexed((idx, f) -> {
                        assertEquals(finalExpected.data().getLong(idx), f.getLong(), epsilon);
                    });
            }
        }else {
            fail("Unexpected DataType: " + dtype);
        }
    }
    
}
