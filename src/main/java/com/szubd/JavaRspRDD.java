package com.szubd;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.RspContext;
import scala.reflect.ClassTag;
import scala.reflect.ClassTag$;

import java.io.Serializable;
import java.lang.reflect.ParameterizedType;
import java.util.ArrayList;
import java.util.Collections;

/**
 * javaRspRDD封装了JavaRDD，可以调用部分RspRDD 的函数
 * 其中LO GO算子进行了java重写，直接对JavaRDD进行操作，不需要进行RDD的转换
 * @param <T>
 */
public class JavaRspRDD<T> implements Serializable {
    private final JavaRDD<T> rdd;

//    RspContext.NewJavaRDDFunc(rdd)
    public JavaRspRDD(JavaRDD<T> rdd) {
        this.rdd = rdd;
    }
    public JavaRDD<T> rdd() {
        return rdd;
    }

    public  <U>JavaRDD<U> LO(Function<T,U> function) {
//        RDD<T> scalaRDD = rdd.rdd();
        // 创建ClassTag
        // 转换函数
        return rdd.mapPartitions(iter -> {
            if (!iter.hasNext()) throw new RuntimeException("Input iterator is empty");
            U result = function.call(iter.next());
            if (iter.hasNext()) throw new RuntimeException("Iterator should have only one element");
            return Collections.singletonList(result).iterator();
        });
    }

//    public <U> JavaRspRDD<U> GO(Function<JavaRDD<T>,JavaRDD<U>> function) throws Exception {
//        JavaRDD<U> resRdd =  function.call(rdd);
//        return new JavaRspRDD<U>(resRdd);
//    }
    public <U> JavaRspRDD<U> GO(Function<JavaRDD<T>, JavaRDD<U>> function) {
        try {
            JavaRDD<U> resRdd = function.call(rdd);
            return new JavaRspRDD<>(resRdd);
        } catch (Exception e) {
            throw new RuntimeException("GO failed", e);
        }
}

    public JavaRDD<T> getSubPartitions(int nums) {
        ParameterizedType type = (ParameterizedType) getClass().getGenericSuperclass();
        Class<T> clazz = (Class<T>) type.getActualTypeArguments()[0];
        ClassTag<T> classTag = ClassTag$.MODULE$.apply(clazz);
        return RspContext.NewJavaRDDFunc(rdd,classTag).getSubPartitions(nums);
    }

//    public JavaRDD<T> getSubPartitions(ArrayList arr) {
//        ParameterizedType type = (ParameterizedType) getClass().getGenericSuperclass();
//        Class<T> clazz = (Class<T>) type.getActualTypeArguments()[0];
//        ClassTag<T> classTag = ClassTag$.MODULE$.apply(clazz);
//        return RspContext.NewJavaRDDFunc(rdd,classTag).getSubPartitions(arr);
//    }

}
