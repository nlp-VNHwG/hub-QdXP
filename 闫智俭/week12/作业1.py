import sqlite3


class ChinookAgent:
    """基于 chinook.db 的简单问答 Agent（NL2SQL）"""

    def __init__(self, db_path: str):
        """
        初始化 Agent，连接数据库并加载表信息。

        Args:
            db_path: chinook.db 文件的路径（相对或绝对）
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # 获取所有表名
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.tables = [row[0] for row in self.cursor.fetchall()]

        # 常用表的中英文映射（可根据需要扩展）
        self.table_map = {
            "员工": "employees",
            "客户": "customers",
            "专辑": "albums",
            "艺术家": "artists",
            "曲目": "tracks",
            "订单": "invoices",
            "订单项": "invoice_items",
            "流派": "genres",
            "媒体类型": "media_types",
            "播放列表": "playlists",
            "播放列表曲目": "playlist_track"
        }

    def answer(self, question: str) -> str:
        """
        根据自然语言问题返回答案。

        Args:
            question: 自然语言问题

        Returns:
            自然语言答案
        """
        # 意图识别与路由
        if '多少张表' in question or '有几个表' in question or '表的总数' in question:
            return self._get_table_count()

        elif '员工' in question and ('记录' in question or '多少' in question or '几条' in question):
            return self._get_record_count('employees')

        elif '客户' in question and '员工' in question and ('个数' in question or '数量' in question):
            return self._get_customer_and_employee_count()

        elif '客户' in question and ('记录' in question or '多少' in question or '几条' in question):
            return self._get_record_count('customers')

        elif '员工' in question and ('个数' in question or '数量' in question):
            return self._get_record_count('employees')

        else:
            # 尝试通过表名映射查询其他表的记录数
            for cn_name, en_name in self.table_map.items():
                if cn_name in question:
                    if '多少' in question or '几条' in question or '记录' in question:
                        return self._get_record_count(en_name)

            return "抱歉，我暂时无法回答这个问题。"

    def _get_table_count(self) -> str:
        """返回数据库中表的总数"""
        count = len(self.tables)
        return f"数据库中共有 {count} 张表。"

    def _get_record_count(self, table_name: str) -> str:
        """返回指定表的记录数"""
        if table_name not in self.tables:
            return f"表 '{table_name}' 不存在。"

        self.cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = self.cursor.fetchone()[0]

        # 获取中文表名用于友好输出
        cn_name = [k for k, v in self.table_map.items() if v == table_name]
        display_name = cn_name[0] if cn_name else table_name

        return f"{display_name} 表中有 {count} 条记录。"

    def _get_customer_and_employee_count(self) -> str:
        """返回客户和员工的数量"""
        self.cursor.execute("SELECT COUNT(*) FROM customers;")
        cust_count = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM employees;")
        emp_count = self.cursor.fetchone()[0]

        return f"客户个数为 {cust_count}，员工个数为 {emp_count}。"

    def close(self):
        """关闭数据库连接"""
        self.conn.close()


if __name__ == '__main__':
    # 使用示例
    agent = ChinookAgent('chinook.db')  # 假设 chinook.db 在当前目录

    test_questions = [
        "数据库中总共有多少张表",
        "员工表中有多少条记录",
        "在数据库中所有客户个数和员工个数分别是多少",
        "客户表有多少条记录",
        "专辑表有几条记录"
    ]

    for q in test_questions:
        print(f"问题：{q}")
        print(f"答案：{agent.answer(q)}\n")

    agent.close()
