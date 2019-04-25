import pandas as pd
import random
import time
import pickle
import pathlib
import os
import tkinter as tk
import numpy as np

'''
 6*6 的站点布局：
-------------------------------------------
| 入口 |  |      |      |      |      |
-------------------------------------------
|      |  |      |      |  |  station1    |
-------------------------------------------
|      |  |      |  |      |      |
-------------------------------------------
| station2|  |      |  |      |      |
-------------------------------------------
|      |  | station3|  |  |      |
-------------------------------------------
|      |      |      |  |      |  |
-------------------------------------------

作者：zhangli
时间：20190415
地点：CQ
'''


class Maze(tk.Tk):
    '''环境类（GUI）'''
    UNIT = 40  # pixels
    MAZE_H = 6  # grid height
    MAZE_W = 6 # grid width
 
    def __init__(self):
        '''初始化'''
        super().__init__()
        self.title('充电站分配模拟')
        h = self.MAZE_H * self.UNIT
        w = self.MAZE_W * self.UNIT
        self.geometry('{0}x{1}'.format(h, w)) #窗口大小
        self.canvas = tk.Canvas(self, bg='white', height=h, width=w)
        # 画网格
        for c in range(0, w, self.UNIT):
            self.canvas.create_line(c, 0, c, h)
        for r in range(0, h, self.UNIT):
            self.canvas.create_line(0, r, w, r)
        # 画陷阱
        self._draw_rect(5, 1, 'yellow')
        self._draw_rect(0, 3, 'yellow')
        self._draw_rect(2, 2, 'yellow')
        #self._draw_rect(5, 5, 'yellow')
        # 画玩家(保存!!)
        self.rect = self._draw_rect(0, 0, 'red')
        self.canvas.pack() # 显示画作！
        
    def _draw_rect(self, x, y, color):
        '''画矩形，  x,y表示横,竖第几个格子'''
        padding = 5 # 内边距5px，参见CSS
        coor = [self.UNIT * x + padding, self.UNIT * y + padding, self.UNIT * (x+1) - padding, self.UNIT * (y+1) - padding]
        return self.canvas.create_rectangle(*coor, fill = color)
 
    def move_to(self, state, delay=0.01):
        '''玩家移动到新位置，根据传入的状态'''
        coor_old = self.canvas.coords(self.rect) # 形如[5.0, 5.0, 35.0, 35.0]（第一个格子左上、右下坐标）
        x, y = state % 6, state // 6 #横竖第几个格子
        padding = 5 # 内边距5px，参见CSS
        coor_new = [self.UNIT * x + padding, self.UNIT * y + padding, self.UNIT * (x+1) - padding, self.UNIT * (y+1) - padding]
        dx_pixels, dy_pixels = coor_new[0] - coor_old[0], coor_new[1] - coor_old[1] # 左上角顶点坐标之差
        self.canvas.move(self.rect, dx_pixels, dy_pixels)
        self.update() # tkinter内置的update!
        time.sleep(delay)


class Agent(object):
    '''个体类'''
    def __init__(self, alpha=0.1, gamma=0.9):
        '''初始化'''
        self.states = range(36)    # 状态集。0~35 共36个状态
        self.actions = list('udlr') # 动作集。上下左右  4个动作
        self.rewards = [10000,0.01,0.01,0.8,0.8, 0.8,
                        0.1,0,0.01,0.01,0.01, 1,                        
                        0.1,0.01,1,10000,10000,10000,
                        0.1,10000,10000,10000,10000,10000,
                        1,10000,10000,10000,10000,10000,10000,
                        10000, 0.02,0.02,0.02,0.02,1,] # 奖励集
        
        self.alpha = alpha
        self.gamma = gamma
        
        self.q_table = pd.DataFrame(data=[[0 for _ in self.actions] for _ in self.states],
                                    index=self.states, 
                                    columns=self.actions)
    
    def save_policy(self):
        '''保存Q table'''
        with open('q_tabletmp.txt', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.q_table, f, pickle.HIGHEST_PROTOCOL)
    
    def load_policy(self):
        '''导入Q table'''
        with open('q_tabletmp.txt', 'rb') as f:
            self.q_table = pickle.load(f)
            
    def write_Record(self,strRecord):
        '''保存Q table'''
        with open('alloresult.txt', 'a') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            f.write(strRecord)       
        f.close()    
    
    def choose_action(self, state, epsilon=0.8):
        '''选择相应的动作。根据当前状态，随机或贪婪，按照参数epsilon'''
        #if (random.uniform(0,1) > epsilon) or ((self.q_table.ix[state] == 0).all()):  # 探索
        if random.uniform(0,1) > epsilon:             # 探索
            action = random.choice(self.get_valid_actions(state))
        else:
            #action = self.q_table.ix[state].idxmax() # 利用 当有多个最大值时，会锁死第一个！
            #action = self.q_table.ix[state].filter(items=self.get_valid_actions(state)).idxmax() # 重大改进！然鹅与上面一样
            s = self.q_table.ix[state].filter(items=self.get_valid_actions(state))
            action = random.choice(s[s==s.min()].index) # 从可能有多个的最小值里面随机选择一个！
        #print(action)    
        return action
    
    def get_q_values(self, state):
        '''取给定状态state的所有Q value'''
        #avgcost=0
        q_values = self.q_table.ix[state, self.get_valid_actions(state)]
        #avgcost+=q_values
        #print("avgcost is :",avgcost/1000)
        return q_values
        
    def update_q_value(self,sumnum, state,  action, next_state_reward, next_state_q_values):
        '''更新Q value，根据贝尔曼方程'''
        atmp=self.alpha * (next_state_reward + self.gamma * next_state_q_values.max() - self.q_table.ix[state, action])
        self.q_table.ix[state, action] += atmp
        sumnum.append(atmp)
        #print("table number is:",sum(sumnum)/len(sumnum) )
    
    def get_valid_actions(self, state):
        '''取当前状态下所有的合法动作'''
        valid_actions = set(self.actions)
        if state % 6 == 5:               # 最后一列，则
            valid_actions -= set(['r'])  # 无向右的动作
        if state % 6 == 0:               # 最前一列，则
            valid_actions -= set(['l'])  # 无向左
        if state // 6 == 5:              # 最后一行，则
            valid_actions -= set(['d'])  # 无向下
        if state // 6 == 0:              # 最前一行，则
            valid_actions -= set(['u'])  # 无向上
        return list(valid_actions)
    
    def get_next_state(self, state, action):
        '''对状态执行动作后，得到下一状态'''
        #u,d,l,r,n = -6,+6,-1,+1,0
        if state % 6 != 5 and action == 'r':    # 除最后一列，皆可向右(+1)
            next_state = state + 1
        elif state % 6 != 0 and action == 'l':  # 除最前一列，皆可向左(-1)
            next_state = state - 1
        elif state // 6 != 5 and action == 'd': # 除最后一行，皆可向下(+2)
            next_state = state + 6
        elif state // 6 != 0 and action == 'u': # 除最前一行，皆可向上(-2)
            next_state = state - 6
        else:
            next_state = state
        return next_state
    
    def learn(self, History,sumnum, env=None, episode=1000, epsilon=0.8,):#history
        '''q-learning算法'''
        avgcost=0
        CAPStation=3
        #,self.states[35]
        termSet=[self.states[11],self.states[18],self.states[14]]
        print('Agent is learning...')
        #print('episode',',','avgcost')
        for i in range(episode):
            current_state = self.states[0]
            
            if env is not None: # 若提供了环境，则重置之！
                #print(i)
                env.move_to(current_state)
                
            while current_state not in termSet:
                current_action = self.choose_action(current_state, epsilon) # 按一定概率，随机或贪婪地选择
                next_state = self.get_next_state(current_state, current_action)
                next_state_reward = self.rewards[next_state]            
                next_state_q_values = self.get_q_values(next_state)
                self.update_q_value(sumnum,current_state, current_action, next_state_reward, next_state_q_values)
                current_state = next_state
                #if next_state != 
                self.rewards[next_state]+=0.2                
                avgcost += next_state_reward
            self.rewards[current_state] += 1
            print(i, "table number is:",np.mean(sumnum) )
            #self.write_Record(str(i)+ "table number is:"+str(np.mean(sumnum))+"\n")
            
            #if current_state==termSet[0]:
              #  History[0]+=1
            #elif current_state == termSet[1]:
             #   History[1] +=1
            #elif current_state == termSet[2]:
             #   History[2] +=1
            #elif current_state== termSet[3]:
             #   History[3] +=1  
            #for i in History:
             #   if i>CAPStation:
              #      current_state=next_state 
               #     agent.learn(History, env, episode=100, epsilon=0.8,)          
                #if next_state not in self.hell_states: # 非陷阱，则往前；否则待在原位
                #    current_state = next_state
                
            if env is not None: # 若提供了环境，则更新之！
                env.move_to(current_state)
            #print(i,',',avgcost)
        #print("the reward is:",self.rewards)    
        print('\nok')
        
    def test(self):
        '''测试agent是否已具有智能'''
        count = 0
        current_state = self.states[0]
        while current_state != self.states[-1]:
            current_action = self.choose_action(current_state, 1.) # 1., 贪婪
            next_state = self.get_next_state(current_state, current_action)
            current_state = next_state
            count += 1
            
            if count > 36:   # 没有在36步之内走出迷宫，则
                return False # 无智能
        
        return True  # 有智能
    
    def play(self,History, env=None, delay=0.5):
        '''玩游戏，使用策略'''
        assert env != None, 'Env must be not None!'
        #,self.states[35]
        termSet=[self.states[11],self.states[18],self.states[14]]
       # History=np.zeros(4)
        
        if not self.test(): # 若尚无智能，则
            if pathlib.Path("q_tabletmp.txt").exists():
                self.load_policy()
            else:
                print("I need to learn before playing this game.")
                self.learn(env, episode=1000, epsilon=0.5)
                self.save_policy()
        
        print('Agent is playing...')
        current_state = self.states[0]
        env.move_to(current_state, delay)
        while current_state not in termSet:
            current_action = self.choose_action(current_state, 1.) # 1., 贪婪
            next_state = self.get_next_state(current_state, current_action)
            current_state = next_state
            self.rewards[next_state]+=0.3  
            env.move_to(current_state, delay)
        if current_state == termSet[0]:
            History[0] +=1
        elif current_state == termSet[1]:
            History[1] +=1
        elif current_state == termSet[2]:
            History[2] +=1
        #elif current_state== termSet[3]:
            #History[3] +=1  
        self.rewards[current_state] += 1     
        print("Result is :", History)  
        self.write_Record(str(History)+"\n")           
        print('\nCongratulations, Agent got it!')


if __name__ == '__main__':
    env = Maze()    # 环境
    agent = Agent() # 个体（智能体）
    History=np.zeros(3)
    sumnum=[]
    #for j in range(96):
    for i in range(20):
        agent.learn(History, sumnum,env, episode=400, epsilon=0.8,) # 先学习history
        agent.save_policy()
        agent.load_policy()
        agent.play(History,env,delay=0.5)                             # 再玩耍
    env.mainloop()
    #env.after(0, agent.learn, env, 1000, 0.8) # 先学
    #env.after(0, agent.save_policy) # 保存所学
    #env.after(0, agent.load_policy) # 导入所学
    #env.after(0, agent.play, env)            # 再玩
   # env.mainloop()
