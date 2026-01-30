export type ApiResponse<T> = {
  ok: boolean;
  data?: T;
  error?: string;
};
export type User = {
  id: string;
  email: string;
  name?: string;
};
