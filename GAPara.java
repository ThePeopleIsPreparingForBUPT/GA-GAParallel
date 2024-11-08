import java.io.*;
import java.util.*;
import java.util.concurrent.CountDownLatch;

import static java.lang.Math.sqrt;

/*
    1、出发城市随机确定，也可以选择其他方法
 */

public class GA {
    private double mutationProbability = 0.08;//变异概率
    private double crossProbability = 0.8;//交叉概率
    private int MAXITER = 300;//最大迭代次数
    private int GROUPSIZE = 200;//种群中个体数目
    private int size = 144;//城市数量
    private ArrayList<ArrayList<Double>> dis = new ArrayList<>();//计算距离
    private ArrayList<ArrayList<Integer>> cities = new ArrayList<>(); //城市信息 ----> (x, y)
    private ArrayList<Integer> bestIndividual = new ArrayList<>();//最优个体
    private double bestFitness = 0;//最优适应度
    private ArrayList<Double> fitness = new ArrayList<>();//种群中每个个体i的适应度
    private double greedyPer = 0.032; //贪心策略生成的个体在种群中所占比例
    private double maxDis = 0; //所有距离之间的最大值
    private ArrayList<ArrayList<Integer>> population = new ArrayList<>();

    private CountDownLatch latch;

    /*
        todo: 1、实现对 anotherBestIndividual 的访问  √
              2、实现对 selfBestIndividual 的修改  √
     */

    public void serviceImpl(ArrayList<Integer> selfBestIndividual, ArrayList<Integer> anotherBestIndividual, ArrayList<ArrayList<Integer>>anoPopu, CountDownLatch latch) {
        this.latch = latch;
        System.out.println("线程" + Thread.currentThread().getName() + "开始工作");
        double[] dist = new double[MAXITER];
        //System.out.println("文件正在读取！");
        getCityInformation();
        //System.out.println("文件读取完毕！");
        //System.out.println("种群正在初始化！");
        ArrayList<ArrayList<Integer>> popultion = initPopultion(selfBestIndividual);
        //System.out.println("种群初始化完毕！");
        for (int i = 0; i < MAXITER; i++) {
            //System.out.println("第" + i + "代:");
            popultion = cross(i, anotherBestIndividual, selfBestIndividual);
            popultion = mutation(selfBestIndividual);
            // printRes();
            dist[i] = 1.0 / bestFitness;

            //进行个体判断
            Set<Integer> set = new HashSet<>();
            for(int v : bestIndividual) {
                set.add(v);
            }
            if(set.size() != size) {
                System.out.println("算法最优个体错误");
                return ;
            }
            // 迭代到一定次数后对部分种群进行进程间交换
            if(i % 20 == 0 && anoPopu.size() == GROUPSIZE) {
                Random random = new Random();
                Integer startIndex = -1, endIndex = -1; // 交换的起始和终止位置
                do{
                    startIndex = random.nextInt(GROUPSIZE);
                    endIndex = random.nextInt(GROUPSIZE);

                }while(startIndex == endIndex); // 确定起始和终止位置
                // 实现种群部分个体交换
                for(int j = startIndex; j <= endIndex; j++) {
                    ArrayList<Integer> temp = population.get(j);
                    population.set(j, anoPopu.get(j));
                    anoPopu.set(j, temp);

//                    if(anoPopu.get(j) != popultion.get(j))
//                        System.out.println("进程间实现了种群互换");
                }
                calPoputionFintess(population, selfBestIndividual);
            }
        }
        draw(dist);
        System.out.println("线程" + Thread.currentThread().getName() + "结束工作");
        latch.countDown();
    }

    public ArrayList<ArrayList<Integer>> test(){
//        getCityInformation();
//        ArrayList<ArrayList<Integer>> popultion = initPopultion();
//        ArrayList<Integer> badIndividualIndex = getBadIndividualIndex(5);
//        System.out.println("坏个体: " + badIndividualIndex);
//        System.out.println("当前适应度:");
//        System.out.println(fitness.size());
//        for(int i = 0; i < fitness.size(); i++) {
//            System.out.println("" + i + ": " + fitness.get(i));
//        }
        return population;
    }

    void draw(double[] distance){
        // 指定要写入的文件路径，如果文件不存在，会自动创建
        String filePath = "example.txt";

        // 使用try-with-resources语句自动管理资源，确保文件在操作完成后正确关闭
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(filePath))) {
            // 写入字符串到文件
            for (int i = 0; i < distance.length; i++) {
                bw.write(""+i);
                if(i != distance.length-1){
                    bw.write(",");
                }
            }
            bw.write("\n");
            for (int i = 0; i < distance.length; i++) {
                bw.write(String.valueOf(distance[i]));
                if(i != distance.length-1){
                    bw.write(",");
                }
            }

            // 注意：在try-with-resources语句块结束时，BufferedWriter会自动关闭
            // 因此，不需要显式调用bw.close();
        } catch (IOException e) {
            // 处理异常，比如打印堆栈跟踪
            e.printStackTrace();
        }

        // 文件写入完成
        //System.out.println("文件写入成功。");
    }

    public void printRes() {
        //System.out.println(bestIndividual);
        System.out.println("最短距离是:" + 1.0 / bestFitness);
    }

    //读取城市信息并计算城市之间的距离
    public void getCityInformation() {
        //读取城市信息
        String fileName = "src\\chn144.txt";
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = br.readLine()) != null) {
                ArrayList<Integer> city = new ArrayList<>();
                String[] s = line.split(" ");
                city.add(Integer.parseInt(s[1]));
                city.add(Integer.parseInt(s[2]));
                cities.add(city);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        //计算距离
        for (int i = 0; i < cities.size(); i++) {
            ArrayList<Double> rowDis = new ArrayList<>();
            for (int j = 0; j < cities.size(); j++) {
                double distance = getDis(cities.get(i), cities.get(j));
                //更新所有距离的最大值
                maxDis = Math.max(maxDis, distance);
                rowDis.add(distance);
            }
            dis.add(rowDis);
        }
    }

    //计算两点之间距离
    public double getDis(ArrayList<Integer> city1, ArrayList<Integer> city2) {
        double distance = 0;
        int x1 = city1.get(0), y1 = city1.get(1), x2 = city2.get(0), y2 = city2.get(1);
        distance = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
        return distance;
    }

    //计算个体适应度 --> 总路程的倒数
    public double calIndividualFitness(ArrayList<Integer> individual) {
        double fitness = 0, sumDist = 0;
        for (int i = 0; i < individual.size() - 1; i++) {
            int start = individual.get(i), end = individual.get(i + 1);
            sumDist += dis.get(start).get(end);
        }
        fitness = 1.0 / sumDist;
        return fitness;
    }

    private void calPartFitnessParaller(ArrayList<ArrayList<Integer>> population, ArrayList<Double> f, CountDownLatch fitnessLatch) {
        for (int i = 0; i < population.size(); i++) {
            f.add(calIndividualFitness(population.get(i)));
        }
        fitnessLatch.countDown();
    }

    //计算种群适应度，并更新最优个体以及最优适应度
    public void calPoputionFintess(ArrayList<ArrayList<Integer>> popultion, ArrayList<Integer> selfBestIndividual) {
        // 并行所需数据定义和获取
        ArrayList<Double> f = new ArrayList<>();// 种群适应值
        ArrayList<Double> f1 = new ArrayList<>(); // 线程 1 计算适应值
        ArrayList<Double> f2 = new ArrayList<>(); // 线程 2 计算适应值
        final ArrayList<ArrayList<Integer>> populationPart1 = new ArrayList<>(); // 线程 1 的种群
        final ArrayList<ArrayList<Integer>> populationPart2 = new ArrayList<>(); // 线程 2 的种群
        for (int i = 0; i < popultion.size(); i++) {
            if(i <= popultion.size() / 2)
                populationPart1.add(popultion.get(i));
            else
                populationPart2.add(popultion.get(i));
        }

        // 实现并行化计算，这里通过 2 个线程实现并需要 同步 操作
        CountDownLatch fitnessLatch = new CountDownLatch(2);
        Runnable r1 = () -> calPartFitnessParaller(populationPart1, f1, fitnessLatch);
        Runnable r2 = () -> calPartFitnessParaller(populationPart2, f2, fitnessLatch);

        Thread thread1 = new Thread(r1);
        Thread thread2 = new Thread(r2);

        thread1.start();
        thread2.start();
        try {
            fitnessLatch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // 错误行检查
        if(f1.size() == 0 || f2.size() == 0){
            System.out.println("适应值并行计算错误");
            System.exit(-1);
        }

        // 并行计算后的收集
        for(int i = 0;i < f1.size();i++) {
            f.add(f1.get(i));
        }
        for(int i = 0;i < f2.size();i++) {
            f.add(f2.get(i));
        }

        fitness = f; // 适应度赋值
        //先确定最优个体的索引，然后更新最优个体和最优适应度
        bestFitness = 0;
        int bestIndividualIndex = 0;
        for (int i = 0; i < fitness.size(); i++) {
            if (fitness.get(i) > bestFitness) {
                bestFitness = fitness.get(i);
                bestIndividualIndex = i;
            }
        }
        bestIndividual = popultion.get(bestIndividualIndex);
        // selfBestIndividual = bestIndividual; // 这一行会导致内存地址的变化，从而违反共享内存规则

        // 实现共享内存策略
        if(selfBestIndividual.size() == 0)
            for(int i = 0; i < bestIndividual.size(); i++) {
                selfBestIndividual.add(bestIndividual.get(i));
            }
        else{
            for(int i = 0; i < bestIndividual.size(); i++) {
                selfBestIndividual.set(i, bestIndividual.get(i));
            }
        }

        bestFitness = fitness.get(bestIndividualIndex);
    }

    //通过贪心策略获得个体
    public void getIndividualByGreedy(int start){
        ArrayList<Integer> individual = new ArrayList<>();
        //individual.add(start);
        greedyDfs(start, start, individual);
    }

    //通过贪心策略获得个体
    public void greedyDfs(int curr, int start, ArrayList<Integer> individual) {
        //System.out.println("population的大小" + population.size());
        if(individual.size() != size){
            individual.add(curr);
            if(individual.size() == size){
                individual.add(start);
                population.add(individual);
                return ;
            }
        }
        int threshold = (int)Math.round(GROUPSIZE * greedyPer);
        if(population.size() == threshold){
            return ;
        }
        ArrayList<ArrayList<Double>> da = new ArrayList<>();
        for(int i = 0; i < dis.get(curr).size(); i++) {
            da.add(new ArrayList<Double>());
            da.get(i).add(dis.get(curr).get(i));
            da.get(i).add(i*1.0);
        }
        Collections.sort(da, (ArrayList<Double> a, ArrayList<Double> b) -> {
            return a.get(0).compareTo(b.get(0));
        });

        for(int i = 0; i < da.size(); i++) {
            int p = (int)Math.round(da.get(i).get(1));
            if(individual.contains(p) == false){
                ArrayList<Integer> temp = new ArrayList<>(individual);
                greedyDfs(p, start, temp);
            }
        }
    }

    //种群初始化 -- > 起点固定
    public ArrayList<ArrayList<Integer>> initPopultion(ArrayList<Integer> selfBestIndividual) {
        Random random = new Random();
        int satrt = 3;
        int greedyInditotal = (int)Math.round(GROUPSIZE * greedyPer);
        getIndividualByGreedy(satrt);
        for (int i = 0; i < GROUPSIZE - greedyInditotal; i++) {
            Set<Integer> set = new HashSet<>();//去重
            ArrayList<Integer> list = new ArrayList<>();//每一个染色
            list.add(satrt);//生成出发点
            set.add(satrt);
            for (int j = 1; j < size; j++) {
                int num = random.nextInt(size);
                while (set.contains(num)) {
                    num = random.nextInt(size);
                }
                set.add(num);
                list.add(num);
            }
            list.add(satrt);
            population.add(list);
        }
        calPoputionFintess(population, selfBestIndividual);
        return population;
    }

    //轮盘赌算法
    public int select() {
        int index = 0;
        double sum = 0;
        for (int i = 0; i < fitness.size(); i++) {
            sum += fitness.get(i);
        }
        ArrayList<Double> probability = new ArrayList<>();
        for (int i = 0; i < fitness.size(); i++) {
            probability.add(fitness.get(i) * 1.0 / sum);
        }
        //System.out.println(probability.size());
        Random random = new Random();
        double threshold = random.nextFloat(), val = 0;
        for (int i = 0; i < probability.size(); i++) {
            val += probability.get(i);
            if (val > threshold) {
                index = i;
                break;
            }
        }
        return index;
    }

    //变异操作，对某一个体的基因位倒换,返回种群
    //1、仅仅交换两位
    //2、交换一个区间 --> use
    public ArrayList<ArrayList<Integer>> mutation(ArrayList<Integer> selfBestIndividual) {
        Random random = new Random();
        for (int i = 0; i < GROUPSIZE; i++) {
            float pro = random.nextFloat();
            if (pro < mutationProbability) {
                //变异操作
                int individualIndex = random.nextInt(GROUPSIZE);//获得变异个体
                //获得变异区间,变异区间为左闭右闭
                int startIndex = -1, endIndex = -1;
                do {
                    startIndex = random.nextInt(size-1)+1;
                    endIndex = random.nextInt(size-1)+1;
                } while (startIndex == endIndex);
                if (startIndex > endIndex) {
                    int t = startIndex;
                    startIndex = endIndex;
                    endIndex = t;
                }

                //System.out.println("当前变异的起始点是" + startIndex + ",终止点是:" + endIndex);

                //开始变异
                int left = startIndex, right = endIndex;
                while (left < right) {
                    //交换两个数
                    int temp = population.get(individualIndex).get(left);
                    population.get(individualIndex).set(left, population.get(individualIndex).get(right));
                    population.get(individualIndex).set(right, temp);
                    left++;
                    right--;
                }
                //System.out.println("变异之后的个体为:" + population.get(individualIndex));
            }
        }
        calPoputionFintess(population, selfBestIndividual);
        return population;
    }

    //变异操作，对某一个体的基因位倒换,返回种群
    //2、交换一个区间 --> use
    //对适应度较低的个体提高变异概率
//    public ArrayList<ArrayList<Integer>> mutation(ArrayList<ArrayList<Integer>> population) {
//        Random random = new Random();
//        ArrayList<Integer> badIndividualIndex = getBadIndividualIndex(5);
//        for (int i = 0; i < GROUPSIZE; i++) {
//            if(badIndividualIndex.contains(i)){
//                mutationProbability = 0.2;
//            }
//            else{
//                mutationProbability = 0.08;
//            }
//            float pro = random.nextFloat();
//            if (pro < mutationProbability) {
//                //变异操作
//                int individualIndex = random.nextInt(GROUPSIZE);//获得变异个体
//
//                //System.out.println("当前变异的个体索引为:" + individualIndex);
//                //System.out.println("当前变异的个体为:" + population.get(individualIndex));
//
//                //获得变异区间,变异区间为左闭右闭
//                int startIndex = -1, endIndex = -1;
//                do {
//                    startIndex = random.nextInt(size-1)+1;
//                    endIndex = random.nextInt(size-1)+1;
//                } while (startIndex == endIndex);
//                if (startIndex > endIndex) {
//                    int t = startIndex;
//                    startIndex = endIndex;
//                    endIndex = t;
//                }
//
//                //System.out.println("当前变异的起始点是" + startIndex + ",终止点是:" + endIndex);
//
//                //开始变异
//                int left = startIndex, right = endIndex;
//                while (left < right) {
//                    //交换两个数
//                    int temp = population.get(individualIndex).get(left);
//                    population.get(individualIndex).set(left, population.get(individualIndex).get(right));
//                    population.get(individualIndex).set(right, temp);
//                    left++;
//                    right--;
//                }
//                //System.out.println("变异之后的个体为:" + population.get(individualIndex));
//            }
//        }
//        calPoputionFintess(population);
//        return population;
//    }

    //混合变异
    //1、对某一个体的基因子序列倒换
    //2、将某一基因子序列放到最后，注意最后一位不能变
    //2、交换一个区间 --> use
    //对适应度较低的个体提高变异概率
//    public ArrayList<ArrayList<Integer>> mutation(ArrayList<ArrayList<Integer>> population) {
//        Random random = new Random();
//        ArrayList<Integer> badIndividualIndex = getBadIndividualIndex(5);
//        for (int i = 0; i < GROUPSIZE; i++) {
//            if(badIndividualIndex.contains(i)){
//                mutationProbability = 0.2;
//            }
//            else{
//                mutationProbability = 0.08;
//            }
//            float pro = random.nextFloat();
//            if (pro < mutationProbability) {
//                //变异操作
//                int individualIndex = random.nextInt(GROUPSIZE);//获得变异个体
//
//                //System.out.println("当前变异的个体索引为:" + individualIndex);
//                //System.out.println("当前变异的个体为:" + population.get(individualIndex));
//
//                //获得变异区间,变异区间为左闭右闭
//                int startIndex = -1, endIndex = -1;
//                do {
//                    startIndex = random.nextInt(size-1)+1;
//                    endIndex = random.nextInt(size-1)+1;
//                } while (startIndex == endIndex);
//                if (startIndex > endIndex) {
//                    int t = startIndex;
//                    startIndex = endIndex;
//                    endIndex = t;
//                }
//
//                //System.out.println("当前变异的起始点是" + startIndex + ",终止点是:" + endIndex);
//
//                //开始第一部分变异 -- > 倒序
//                int left = startIndex, right = endIndex;
//                while (left < right) {
//                    //交换两个数
//                    int temp = population.get(individualIndex).get(left);
//                    population.get(individualIndex).set(left, population.get(individualIndex).get(right));
//                    population.get(individualIndex).set(right, temp);
//                    left++;
//                    right--;
//                }
//
//                //开始第二部分变异 --> 进行插入变异
//                do {
//                    startIndex = random.nextInt(size-1)+1;
//                    endIndex = random.nextInt(size-1)+1;
//                } while (startIndex == endIndex);
//                if (startIndex > endIndex) {
//                    int t = startIndex;
//                    startIndex = endIndex;
//                    endIndex = t;
//                }
//                ArrayList<Integer> newIndividual = new ArrayList<>();
//                for(int j = 0;j < population.get(individualIndex).size() - 1;j ++ ){
//                    if(j < startIndex || j > endIndex)
//                        newIndividual.add(population.get(individualIndex).get(j));
//                }
//                for(int j = startIndex; j <= endIndex; j ++ ){
//                    newIndividual.add(population.get(individualIndex).get(j));
//                }
//                newIndividual.add(newIndividual.get(0));
//                population.set(individualIndex, newIndividual);
//                //System.out.println("变异之后的个体为:" + population.get(individualIndex));
//            }
//        }
//        calPoputionFintess(population);
//        return population;
//    }

    //OX交叉
    public ArrayList<ArrayList<Integer>> cross(int iter, ArrayList<Integer> anotherBestIndividual, ArrayList<Integer> selfBestIndividual) {
        Random random = new Random();
        for (int i = 0; i < GROUPSIZE; i++) {
            double pro = random.nextDouble();
            if (pro < crossProbability) {
                // 这里对交叉操作的两个父代进行操作
                // 一部分来自轮盘赌，另一部分来自上一个进程的最优父代
                // todo: 这里存在一个问题，那就是两个进程的起点和终点可能不一致， 但查阅种群初始化函数，发现起始点固定了
                int sel = select();
                ArrayList<Integer> individual = population.get(sel);
                if(iter % 20 == 0 && anotherBestIndividual != null && anotherBestIndividual.size() > 0){
                    individual = anotherBestIndividual; // todo: 这里没有进行深度拷贝， 请注意
                }
//                if(anotherBestIndividual.size() != 0){
//                    System.out.println("线程间实现了通信");
//                }
                //进行交叉操作，一个来自轮盘赌，另一个来自当前最优个体
                //获得交叉区间,变异区间为左闭右闭
                int startIndex = -1, endIndex = -1;
                do {
                    startIndex = random.nextInt(size-1)+1;
                    endIndex = random.nextInt(size-1)+1;//
                } while (startIndex == endIndex);
                if (startIndex > endIndex) {
                    int t = startIndex;
                    startIndex = endIndex;
                    endIndex = t;
                }
                //进行OX交叉
                int cnt = 1;
                ArrayList<Integer> child = new ArrayList<>();
                child.add(bestIndividual.get(0));
                for (int j = 1; j < size; j++) {
                    child.add(-2);
                }
                child.add(bestIndividual.get(0));
                for (int j = startIndex; j <= endIndex; j++) {
                    child.set(j, bestIndividual.get(j));
                }
                for(int j = 1; j < size; j++) {
                    if(child.contains(individual.get(j))) continue;
                    else{
                        if(cnt == startIndex)   cnt = endIndex + 1;
                        if(cnt >= size) break;
                        child.set(cnt, individual.get(j));
                        cnt++;
                    }
                }
                //交叉结束 -- > 替换轮盘赌选择的个体
                if (calIndividualFitness(child) > calIndividualFitness(individual)) {
                    population.set(sel, child);
                }
                calPoputionFintess(population, selfBestIndividual);
            }
        }
        return population;
    }

    //获得坏个体索引
    public ArrayList<Integer> getBadIndividualIndex(int cnt){
        ArrayList<Integer> badIndiIndex = new ArrayList<>();
        ArrayList<ArrayList<Double>> fit = new ArrayList<>();
        for (int i = 0; i < fitness.size(); i++) {
            fit.add(new ArrayList<>());
            fit.get(i).add(fitness.get(i));
            fit.get(i).add(i*1.0);
        }
        Collections.sort(fit, (ArrayList<Double> a, ArrayList<Double> b) ->{
            return a.get(0).compareTo(b.get(0));
        });
        //返回最后{cnt}个个体对应的索引
        for (int i = 0; i < cnt; i++) {
            badIndiIndex.add((int) Math.round(fit.get(i).get(1)));
        }
        return badIndiIndex;
    }

    public Double getMinDis(){
        return 1 / bestFitness;
    }

    public ArrayList<Integer> getBestIndividual() {
        return bestIndividual;
    }

    public void setBestIndividual(ArrayList<Integer> bestIndividual) {
        this.bestIndividual = bestIndividual;
    }

    public ArrayList<ArrayList<Integer>> getPopulation() {
        return population;
    }

    public void setPopulation(ArrayList<ArrayList<Integer>> population) {
        this.population = population;
    }
}
